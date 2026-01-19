import numpy as np
import pandas as pd


# 彻底删掉 scipy.stats，它是报错的源头
def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.sort_values("Time").reset_index(drop=True)

    # 获取预测步长 (24小时 = 144个10分钟点)
    h = config["forecast_config"].get("look_ahead_steps", 144)

    # --- 核心功能：未来风速预测 ---
    if "WindSpeed" in df.columns:
        # 训练时：将未来风速拉回当前行。预测时：这里填入的是气象预报风速。
        df[f"future_ws_{h}"] = df["WindSpeed"].shift(-h)

        # 增加物理项 (风能与风速立方成正比)
        df["ws_squared"] = df["WindSpeed"] ** 2
        df["ws_cubed"] = df["WindSpeed"] ** 3
        df["ws_diff_1"] = df["WindSpeed"].diff(1)
        df["ws_rolling_std_6"] = df["WindSpeed"].rolling(window=6).std()

    # 滞后特征 (Lag Features)
    lags = [1, 6, 12, 144]
    for lag in lags:
        if "ActivePower" in df.columns:
            df[f"power_lag_{lag}"] = df["ActivePower"].shift(lag)

    # 滚动特征 - 使用 Pandas 原生 kurt()，不调 scipy
    if "ActivePower" in df.columns:
        df["power_diff_1"] = df["ActivePower"].diff(1)
        window_sizes = [6, 12]
        for w in window_sizes:
            rolling = df["ActivePower"].rolling(window=w, min_periods=1)
            df[f"power_rolling_mean_{w}"] = rolling.mean()
            df[f"power_rolling_std_{w}"] = rolling.std()
            # 关键：用 pandas 自带的峰度函数，绝对不会报错
            df[f"power_rolling_kurt_{w}"] = rolling.kurt().fillna(0)

    # 时间周期特征
    df["hour_sin"] = np.sin(2 * np.pi * df["Time"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Time"].dt.hour / 24)

    return df