import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.features import build_features
from src.trainer import Trainer
from src.data_loader import load_and_clean_data
from src.noise_processing import noise_handling_pipeline

logging.basicConfig(level=logging.INFO)


def run_pipeline(config_path: str):
    # 1. 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 载入并清洗数据
    df = load_and_clean_data(config["paths"]["raw_data"], config)

    # 3. 噪声处理
    df = noise_handling_pipeline(df, config)

    # 4. 构建特征
    df_feat = build_features(df, config)

    # 5. 构建预测目标
    h = config["forecast_config"].get("look_ahead_steps", 6)
    df_feat["target_future"] = df_feat["ActivePower"].shift(-h)

    # 6. 剔除因 shift 或长缺失导致的 NaN 行
    df_feat = df_feat.dropna().reset_index(drop=True)
    logging.info(f"Final data rows for training/test: {len(df_feat)}")

    # 7. 特征选择
    from src.selector import select_features
    features = select_features(df_feat, config)
    logging.info(f"Features used: {features}")

    # 8. 划分数据集
    test_size = config["forecast_config"].get("test_size", 0.2)
    split_idx = int(len(df_feat) * (1 - test_size))

    train_df = df_feat.iloc[:split_idx]
    test_df = df_feat.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df["target_future"]
    X_test, y_test = test_df[features], test_df["target_future"]

    # 9. 训练与评估
    trainer = Trainer(config["model_params"], config["forecast_config"]["capacity_kw"])
    model = trainer.train(X_train, y_train, X_test, y_test)

    r2, mae, acc = trainer.evaluate(model, X_test, y_test)
    logging.info(f"FINAL RESULT >> Accuracy={acc:.2f}% | R2={r2:.4f} | MAE={mae:.2f}")

    # 10. 绘图：预测对比 + 特征重要度
    y_pred = model.predict(X_test)
    plot_enhanced_results(y_test, y_pred, model, features)


def plot_enhanced_results(y_test, y_pred, model, features):
    """
    绘制预测结果对比图图 和 特征重要度排行
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(15, 12))

    # 子图1：时序对比
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    show_len = min(len(y_test), 500)
    ax1.plot(y_test.values[:show_len], label='实际功率', color='#1f77b4', alpha=0.8, linewidth=2)
    ax1.plot(y_pred[:show_len], label='预测功率 (已知未来风速)', color='#ff7f0e', linestyle='--', alpha=0.9)
    ax1.set_title(f"功率预测对比 (前 {show_len} 个点)", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：残差图
    residuals = y_test.values - y_pred

    ax2 = plt.subplot2grid((3, 1), (2, 0))
    # 绘制残差散点图
    ax2.scatter(y_pred, residuals, alpha=0.5, color='#d62728', s=10)
    # 绘制 y=0 的基准线
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)

    ax2.set_xlabel('预测功率 (kW)')
    ax2.set_ylabel('残差 (实际 - 预测)')
    ax2.set_title("残差分布图", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    logging.info("Drawing figure...")
    plt.show()


if __name__ == "__main__":
    run_pipeline("config/config.yaml")




