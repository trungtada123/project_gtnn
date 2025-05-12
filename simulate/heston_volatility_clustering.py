
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def heston_simulate(S0, v0, mu, kappa, theta, sigma, rho, T, N, M):
    dt = T / N
    S = np.zeros((M, N + 1))
    v = np.zeros((M, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(1, N + 1):
        Z1 = np.random.normal(0, 1, M)
        Z2 = np.random.normal(0, 1, M)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

        v[:, t] = np.maximum(v[:, t-1] + kappa * (theta - v[:, t-1]) * dt +
                             sigma * np.sqrt(np.maximum(v[:, t-1], 0)) * np.sqrt(dt) * W2, 0)
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * v[:, t-1]) * dt +
                                     np.sqrt(np.maximum(v[:, t-1], 0)) * np.sqrt(dt) * W1)
    return S, v

def price_european_call(S_paths, K, r, T):
    payoff = np.maximum(S_paths[:, -1] - K, 0)
    return np.exp(-r * T) * np.mean(payoff)


# Các bộ tham số khác nhau (volatility clustering)
param_sets = [
    {"kappa": 1.0, "theta": 0.02, "sigma": 0.3, "label": "Low mean-rev"},
    {"kappa": 3.0, "theta": 0.04, "sigma": 0.5, "label": "Base case"},
    {"kappa": 5.0, "theta": 0.06, "sigma": 0.6, "label": "Strong mean-rev"},
    {"kappa": 2.0, "theta": 0.04, "sigma": 0.9, "label": "High vol-of-vol"},
    {"kappa": 2.0, "theta": 0.04, "sigma": 0.2, "label": "Low vol-of-vol"},
]

# Tham số cố định
S0 = 100
v0 = 0.04
mu = 0.05
rho = -0.7
T = 1.0
N = 252
M = 1000
K = 100
r = 0.01

results = []
simulated_prices = {}

for params in param_sets:
    S, v = heston_simulate(S0, v0, mu, params["kappa"], params["theta"], params["sigma"], rho, T, N, M)
    price = price_european_call(S, K, r, T)
    results.append({
        "Label": params["label"],
        "kappa": params["kappa"],
        "theta": params["theta"],
        "sigma": params["sigma"],
        "Call Price": round(price, 4)
    })
    simulated_prices[params["label"]] = S[:5]  # 5 đường đầu tiên để vẽ

# Chuyển kết quả thành DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Vẽ biểu đồ so sánh các đường giá
time = np.linspace(0, T, N + 1)
plt.figure(figsize=(10, 6))
for label, prices in simulated_prices.items():
    for i in range(prices.shape[0]):
        plt.plot(time, prices[i], label=label if i == 0 else "", alpha=0.7)
plt.title("So sánh các đường giá $S_t$ với các bộ tham số khác nhau (Volatility Clustering)")
plt.xlabel("Thời gian")
plt.ylabel("Giá tài sản $S_t$")
plt.legend()
plt.tight_layout()
plt.show()
