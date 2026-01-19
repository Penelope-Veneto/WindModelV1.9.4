# 风电预测 v1.9.4

**Wind Power Forecasting System**

基于LightGBM的风电发电量预测模型。通过输入未来24小时内风速自动预测风力发电机发电量。

by Yutong Zhou

---

## 项目结构

```
WindModelV1.9.3/
│
├── main.py              # 主训练程序
├── README.md           # 使用说明
├── config/
│    └── config.yaml        #配置文件
│
├── data/
│   └── raw/                # 原始数据文件夹
│      └─  turbine1.csv    # 数据文件1
│
└─ models/                 # 模型文件夹
    ├── lightgbm_model.txt  # 训练好的模型
    ├── scaler.pkl          # 特征缩放器
    └─ feature_info.json   # 特征信息

```

---

##  快速开始

### 1. 安装依赖
安装numpy,lightGBM等依赖文件

### 2. 准备数据

将你的CSV数据文件放入 `data/raw/` 文件夹：

```bash
data/
└── raw/
    ├── turbine_data_2023.csv
    ├── turbine_data_2024.csv
    └── ...
```

**数据要求：**
- CSV格式
- 包含发电量列（如 `ActivePower`, `power`, `generation` 等）
- 包含时间列（如 `date`, `timestamp`, `Date/Time` 等）

### 3. 修改配置

编辑 `config.yaml` 文件，设置关键参数：

```yaml
# 变量选择配置
feature_selection:
  mode: "manual"            #top_n or manual
  manual_features:
    - 手动输入变量名

top_n_count: 8 #mode为top_n时使用，选相关性最高的n个特征

# 数据划分比例
forecast_config:
  eg1: 1
  eg2: 2      
  test_size: 0.2     # 测试集20%
```

### 4. 运行训练

```bash
python main.py
```

### 5. 查看结果

训练完成后，查看结果：
- **评估指标**：控制台输出 MAE、准确度、R²
- **可视化图表**：代码生成图片

---

## 更新日志

### v1.9.4 (2026-01-19)
-  改进了配置文件，使其修改更方便
-  两种特征选择模式（manual/top_n） 
-  完整的评估指标（MAE/R²/通过MAPE计算的准确度）
-  自动可视化输出
-  优化的代码结构

