import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class HestonImportanceSampling:
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r, T):
        """
        Khởi tạo các tham số cho mô hình Heston
        
        Tham số:
        -----------
        S0 : float
            Giá tài sản ban đầu
        v0 : float
            Biến động (variance) ban đầu
        kappa : float
            Tốc độ hồi quy trung bình của biến động
        theta : float
            Mức biến động dài hạn
        sigma : float
            Độ biến động của biến động (volatility of variance)
        rho : float
            Hệ số tương quan giữa quá trình tài sản và quá trình biến động
        r : float
            Lãi suất phi rủi ro
        T : float
            Thời gian đáo hạn
        """
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
        self.T = T
    
    def _generate_paths_standard(self, K, n_steps, n_paths, antithetic=True):
        """Sinh các đường dẫn sử dụng Monte Carlo tiêu chuẩn"""
        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Khởi tạo mảng
        S = np.zeros((n_paths, n_steps+1))
        v = np.zeros((n_paths, n_steps+1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        # Sinh số ngẫu nhiên tương quan
        if antithetic:
            Z1 = np.random.normal(size=(n_paths//2, n_steps))
            Z2 = np.random.normal(size=(n_paths//2, n_steps))
            Z1 = np.vstack((Z1, -Z1))  # Biến đối nghịch (Antithetic variates)
            Z2 = np.vstack((Z2, -Z2))  # Biến đối nghịch (Antithetic variates)
        else:
            Z1 = np.random.normal(size=(n_paths, n_steps))
            Z2 = np.random.normal(size=(n_paths, n_steps))
            
        # Chuyển động Brown tương quan
        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)
        
        # Mô phỏng các đường dẫn
        for i in range(n_steps):
            # Cắt ngưỡng đầy đủ cho variance để đảm bảo giá trị dương
            v_pos = np.maximum(v[:, i], 0)
            
            # Cập nhật variance
            v[:, i+1] = v[:, i] + self.kappa * (self.theta - v_pos) * dt + \
                         self.sigma * np.sqrt(v_pos) * dW2[:, i]
            
            # Cập nhật giá tài sản
            S[:, i+1] = S[:, i] * np.exp((self.r - 0.5 * v_pos) * dt + 
                                         np.sqrt(v_pos) * dW1[:, i])
        
        return S, v
    
    def _calculate_optimal_shift(self, K):
        """Tính toán tham số dịch chuyển tối ưu dựa trên độ moneyness của quyền chọn"""
        # Độ moneyness của quyền chọn
        moneyness = self.S0 / K
        
        # Hàm heuristic đơn giản để xác định tham số dịch chuyển dựa trên moneyness
        if moneyness < 0.9:  # Deep OTM (Ngoài tiền sâu)
            shift_S = 0.6
            shift_v = 0.3
        elif moneyness < 1.0:  # OTM (Ngoài tiền)
            shift_S = 0.4
            shift_v = 0.2
        elif moneyness == 1.0:  # ATM (Tại tiền)
            shift_S = 0.2
            shift_v = 0.1
        elif moneyness < 1.1:  # ITM (Trong tiền)
            shift_S = 0.1
            shift_v = 0.05
        else:  # Deep ITM (Trong tiền sâu)
            shift_S = 0.0
            shift_v = 0.0
            
        return shift_S, shift_v
    
    def _generate_paths_IS(self, K, n_steps, n_paths):
        """Sinh các đường dẫn sử dụng phương pháp Importance Sampling với dịch chuyển tối ưu"""
        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Lấy tham số dịch chuyển tối ưu
        shift_S, shift_v = self._calculate_optimal_shift(K)
        
        # Khởi tạo mảng
        S = np.zeros((n_paths, n_steps+1))
        v = np.zeros((n_paths, n_steps+1))
        L = np.ones(n_paths)  # Tỷ số likelihood
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        # Sinh số ngẫu nhiên phân phối chuẩn
        Z1 = np.random.normal(size=(n_paths, n_steps))
        Z2 = np.random.normal(size=(n_paths, n_steps))
            
        # Chuyển động Brown tương quan không có IS
        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)
        
        # Mô phỏng các đường dẫn
        for i in range(n_steps):
            # Cắt ngưỡng đầy đủ cho variance để đảm bảo giá trị dương
            v_pos = np.maximum(v[:, i], 0)
            sqrt_v_pos = np.sqrt(v_pos)
            
            # Tính toán các thành phần dịch chuyển
            shift_term_S = shift_S * sqrt_v_pos
            shift_term_v = shift_v * sqrt_v_pos
            
            # Cập nhật tỷ số likelihood
            L *= np.exp(-shift_term_S * dW1[:, i]/sqrt_dt - shift_term_v * dW2[:, i]/sqrt_dt - 
                         0.5 * (shift_term_S**2 + shift_term_v**2) * dt)
            
            # Cập nhật variance với sự dịch chuyển Importance Sampling
            v[:, i+1] = v[:, i] + self.kappa * (self.theta - v_pos) * dt + \
                         self.sigma * sqrt_v_pos * dW2[:, i] + \
                         shift_term_v * self.sigma * dt
            
            # Cập nhật giá tài sản với sự dịch chuyển Importance Sampling
            S[:, i+1] = S[:, i] * np.exp((self.r - 0.5 * v_pos) * dt + 
                                         sqrt_v_pos * dW1[:, i] + 
                                         shift_term_S * dt)
        
        return S, v, L
    
    def price_european_call(self, K, n_steps, n_paths, method='standard'):
        """Định giá quyền chọn mua kiểu Âu với các phương pháp khác nhau"""
        if method == 'standard':
            S, _ = self._generate_paths_standard(K, n_steps, n_paths)
            payoffs = np.maximum(S[:, -1] - K, 0)
            price = np.exp(-self.r * self.T) * np.mean(payoffs)
            stderr = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(n_paths)
        elif method == 'IS':
            S, _, L = self._generate_paths_IS(K, n_steps, n_paths)
            payoffs = np.maximum(S[:, -1] - K, 0) * L
            price = np.exp(-self.r * self.T) * np.mean(payoffs)
            stderr = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(n_paths)
        else:
            raise ValueError("Phương pháp phải là 'standard' hoặc 'IS'")
            
        return price, stderr
    
    def price_european_put(self, K, n_steps, n_paths, method='standard'):
        """Định giá quyền chọn bán kiểu Âu với các phương pháp khác nhau"""
        if method == 'standard':
            S, _ = self._generate_paths_standard(K, n_steps, n_paths)
            payoffs = np.maximum(K - S[:, -1], 0)
            price = np.exp(-self.r * self.T) * np.mean(payoffs)
            stderr = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(n_paths)
        elif method == 'IS':
            S, _, L = self._generate_paths_IS(K, n_steps, n_paths)
            payoffs = np.maximum(K - S[:, -1], 0) * L
            price = np.exp(-self.r * self.T) * np.mean(payoffs)
            stderr = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(n_paths)
        else:
            raise ValueError("Phương pháp phải là 'standard' hoặc 'IS'")
            
        return price, stderr
    
    def compute_variance_reduction(self, K, n_steps, n_paths):
        """Tính toán hệ số giảm phương sai đạt được bởi IS"""
        # Tính phương sai cho MC tiêu chuẩn
        S_std, _ = self._generate_paths_standard(K, n_steps, n_paths)
        payoffs_std = np.maximum(S_std[:, -1] - K, 0)
        var_std = np.var(payoffs_std)
        
        # Tính phương sai cho IS
        S_is, _, L = self._generate_paths_IS(K, n_steps, n_paths)
        payoffs_is = np.maximum(S_is[:, -1] - K, 0) * L
        var_is = np.var(payoffs_is)
        
        # Tính hệ số giảm phương sai
        if var_is > 0:
            var_reduction = var_std / var_is
        else:
            var_reduction = float('inf')
            
        return var_reduction