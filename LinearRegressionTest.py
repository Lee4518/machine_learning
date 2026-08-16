import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. 生成少量样本数据 ---
# 设置随机种子，使结果可复现
np.random.seed(42)

# 生成特征 X (0 到 10 之间的 20 个点)
X = np.linspace(0, 10, 20).reshape(-1, 1)

# 真实关系: y = 2.5 * X + 1.8，并加入少量随机噪声
true_slope = 2.5
true_intercept = 1.8
noise = np.random.normal(0, 1.5, X.shape)  # 均值0，标准差1.5的高斯噪声
y = true_slope * X + true_intercept + noise

# --- 2. 创建并训练线性回归模型 ---
model = LinearRegression()
model.fit(X, y)

# 获取训练后的参数
slope = model.coef_[0]
intercept = model.intercept_
print(f"训练得到的斜率: {slope:.4f}")
print(f"训练得到的截距: {intercept:.4f}")
print(f"真实斜率: {true_slope}, 真实截距: {true_intercept}")

# --- 3. 使用模型进行预测 ---
y_pred = model.predict(X)

# --- 4. 评估模型性能 ---
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"\n均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R^2): {r2:.4f}")

# --- 5. 可视化结果 ---
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='实际数据点')
plt.plot(X, y_pred, color='red', linewidth=2, label='回归直线')
plt.xlabel('特征 X')
plt.ylabel('目标 y')
plt.title('线性回归预测 (少量数据)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()