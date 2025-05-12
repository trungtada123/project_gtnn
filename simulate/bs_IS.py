
import numpy as np

def monte_carlo_option_pricing_IS(S0, K, T, r, sigma, N=10000, option_type="call", shift=0.2):
    """
    Định giá quyền chọn châu Âu bằng phương pháp Monte Carlo + Importance Sampling.

    Tham số:
    - S0: giá tài sản ban đầu
    - K: giá thực hiện
    - T: thời gian đáo hạn
    - r: lãi suất phi rủi ro
    - sigma: độ biến động
    - N: số lượng mô phỏng
    - option_type: "call" hoặc "put"
    - shift: hệ số dịch chuyển phân phối (importance sampling)

    Trả về:
    - Giá quyền chọn ước lượng theo IS
    """
    # Dịch chuyển drift trong phân phối chuẩn (importance sampling)
    Z_tilde = np.random.randn(N) + shift  # Lấy mẫu từ q(x)
    S_T_tilde = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_tilde)

    # Tính trọng số likelihood ratio: p(Z) / q(Z)
    w = np.exp(-shift * Z_tilde + 0.5 * shift**2)

    # Tính payoff
    if option_type == "call":
        payoff = np.maximum(S_T_tilde - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - S_T_tilde, 0)
    else:
        raise ValueError("Option type phải là 'call' hoặc 'put'")

    weighted_payoff = payoff * w
    price = np.exp(-r * T) * np.mean(weighted_payoff)
    return price


# Ví dụ sử dụng
if __name__ == "__main__":
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    N = 10000

    # Monte Carlo gốc
    Z = np.random.randn(N)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    call_mc = np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))

    # IS
    call_is = monte_carlo_option_pricing_IS(S0, K, T, r, sigma, N=N, option_type="call", shift=0.2)

    print(f"Monte Carlo thông thường (Call): {call_mc:.4f}")
    print(f"Monte Carlo + IS (Call):        {call_is:.4f}")
