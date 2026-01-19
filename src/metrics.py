import numpy as np

def pointwise_accuracy(y_true, y_pred, capacity_kw):
    y_true = np.asarray(y_true)
    y_pred = np.maximum(0, np.asarray(y_pred))

    # 手动计算 MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # 手动计算 R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # 你的核心精度逻辑
    denom = np.where(y_true == 0, capacity_kw, y_true)
    point_acc = 1 - (np.abs(y_true - y_pred) / denom)
    point_acc_clipped = np.clip(point_acc, 0, 1)
    final_acc = point_acc_clipped.mean() * 100

    return {
        "Accuracy": final_acc,
        "MAE": mae,
        "R2": r2
    }