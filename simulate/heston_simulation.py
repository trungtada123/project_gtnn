
import numpy as np

def heston_simulate(S0, v0, mu, kappa, theta, sigma, rho, T, N, M):
    """
    Mô phỏng mô hình Heston bằng phương pháp Euler–Maruyama.

    Tham số:
    - S0: giá ban đầu
    - v0: phương sai ban đầu
    - mu: lợi suất kỳ vọng
    - kappa: tốc độ hồi về trung bình
    - theta: phương sai dài hạn
    - sigma: độ biến động phương sai
    - rho: hệ số tương quan giữa 2 nhiễu
    - T: thời gian mô phỏng (năm)
    - N: số bước thời gian
    - M: số đường mô phỏng

    Trả về:
    - S: ma trận giá cổ phiếu (M x (N+1))
    - v: ma trận phương sai (M x (N+1))
    """
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
    """
    Định giá quyền chọn mua châu Âu từ các đường mô phỏng giá tài sản.

    Tham số:
    - S_paths: ma trận mô phỏng giá tài sản (M x N+1)
    - K: giá thực hiện
    - r: lãi suất phi rủi ro
    - T: thời gian đáo hạn

    Trả về:
    - Giá quyền chọn mua
    """
    payoff = np.maximum(S_paths[:, -1] - K, 0)
    return np.exp(-r * T) * np.mean(payoff)
