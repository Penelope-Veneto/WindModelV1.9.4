import pandas as pd
import numpy as np
import logging


def load_and_clean_data(path, config):
    logging.info(f"Loading data: {path}")
    df = pd.read_csv(path, encoding="utf-8")

    # 1. 统一时间列名
    df = df.rename(columns={df.columns[0]: "Time"})
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time")

    # 2. 自动检测频率并对齐时间轴
    time_diffs = df["Time"].diff().dt.total_seconds() / 60
    median_freq = time_diffs.median()
    if pd.isna(median_freq) or median_freq <= 0:
        median_freq = 10  # 默认10分钟

    full_range = pd.date_range(start=df["Time"].min(), end=df["Time"].max(), freq=f'{int(median_freq)}min')
    df = df.set_index("Time").reindex(full_range).rename_axis("Time").reset_index()

    # 3. 连续缺失处理 (12个点 = 2小时)
    threshold_rows = 12
    for col in df.select_dtypes(include=np.number).columns:
        if col == "Time": continue

        is_na = df[col].isna()
        na_groups = (is_na != is_na.shift()).cumsum()[is_na]

        # 先插值以便后续计算 Lag 特征
        df[col] = df[col].interpolate(method='linear')

        # 超过2小时的空洞，插值后依然抹除为 NaN，确保不参加训练
        if not na_groups.empty:
            group_counts = na_groups.value_counts()
            large_gap_ids = group_counts[group_counts > threshold_rows].index
            mask_large = na_groups.isin(large_gap_ids)
            df.loc[mask_large.index[mask_large], col] = np.nan

    # 4. 排除黑名单
    exclude = config.get("exclude_columns", [])
    df = df.drop(columns=[c for c in exclude if c in df.columns and c != "Time"])

    return df