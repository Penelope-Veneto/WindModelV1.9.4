import numpy as np
import pandas as pd


def remove_outliers_iqr(df, col, k=1.5):
    """使用IQR方法剔除异常值"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
    df.loc[~mask, col] = np.nan
    # 使用插值修复被剔除的点
    df[col] = df[col].interpolate(method='linear').bfill().ffill()
    return df


def detect_spikes(df, col, threshold):
    diffs = df[col].diff().abs()
    spikes = diffs > threshold
    df.loc[spikes, col] = np.nan
    df[col] = df[col].interpolate().bfill().ffill()
    return df


def noise_handling_pipeline(df, config):
    noise_cfg = config.get("noise_processing", {})
    if not noise_cfg.get("enabled", False):
        return df

    # 1. 物理约束修复：针对残差图中低功率区发散的问题
    # 当风速极低时，强制功率为0，消除模型在起步阶段的毛刺预测
    if "WindSpeed" in df.columns and "ActivePower" in df.columns:
        df.loc[df["WindSpeed"] < 0, "ActivePower"] = 0

    target = "ActivePower"
    if target in df.columns:
        df = remove_outliers_iqr(df, target, k=noise_cfg.get("iqr_k", 1.5))
        df = detect_spikes(df, target, threshold=noise_cfg.get("spike_threshold", 300))

    return df