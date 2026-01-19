# src/selector.py
import logging

FORBIDDEN = {
    "Time",
    "ActivePower",
    "target_future",
    "WindSpeed"   # 原始列不直接用
}

def select_features(df_train, config):
    cfg = config.get("feature_selection", {})
    mode = cfg.get("mode", "manual")

    if mode == "manual":
        feats = [
            f for f in cfg.get("manual_features", [])
            if f in df_train.columns
        ]
        logging.info(f"Manual features selected: {feats}")
        return feats

    if mode == "top_n":
        n = cfg.get("top_n_count", 5)
        corr = (
            df_train
            .drop(columns=FORBIDDEN, errors="ignore")
            .corr()["target_future"]
            .abs()
            .sort_values(ascending=False)
        )
        feats = corr.head(n).index.tolist()
        logging.info(f"Top-{n} correlation features: {feats}")
        return feats

    return [c for c in df_train.columns if c not in FORBIDDEN]
