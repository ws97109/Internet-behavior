import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats import chi2

class PCATestAnalyzer:
    def __init__(self, data_path):
        """初始化 PCA 分析器"""
        self.df = pd.read_csv(data_path)
        self.X = None
        self.attitude_cols = None
        self.attitude_groups = None
        
    def prepare_data(self):
        """準備數據"""
        self.attitude_groups = {
            'behavior_obs': [f'q22_0{i}_1' for i in range(1, 6)],  # 觀察到的網路行為
            'personal_act': [f'q23_0{i}_1' for i in range(1, 6)],  # 個人網路行為
            'acceptance': [f'q25_0{i}_1' for i in range(1, 5)],    # 行為接受度
            'influence': [f'q26_0{i}_1' for i in range(1, 4)]      # 影響評估
        }
        
        self.attitude_cols = [col for group in self.attitude_groups.values() for col in group]
        self.X = self.df[self.attitude_cols].dropna()
        
    def perform_kmo_test(self):
        """執行 KMO 檢定"""
        try:
            # 使用相關矩陣進行KMO檢定
            kmo_all, kmo_model = calculate_kmo(self.X)
            
            print("\nKMO 檢定結果:")
            print("=" * 50)
            # 重要：將numpy array轉換為純數值
            kmo_value = float(kmo_all.mean())  # 取平均值轉換為純數值
            print(f"KMO 值: {kmo_value:.3f}")
            
            # KMO 評價標準
            if kmo_value >= 0.9:
                print("評價: 極佳 (Marvelous)")
            elif kmo_value >= 0.8:
                print("評價: 優良 (Meritorious)")
            elif kmo_value >= 0.7:
                print("評價: 中等 (Middling)")
            elif kmo_value >= 0.6:
                print("評價: 普通 (Mediocre)")
            elif kmo_value >= 0.5:
                print("評價: 不佳 (Miserable)")
            else:
                print("評價: 不適合 (Unacceptable)")
            
            return kmo_value, kmo_model
            
        except Exception as e:
            print(f"KMO 檢定過程中發生錯誤: {str(e)}")
            return None, None
        
    def perform_bartlett_test(self):
        """執行 Bartlett's 球形檢定"""
        try:
            correlation_matrix = self.X.corr()
            n = len(self.X)
            p = len(self.X.columns)
            chi_square = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(correlation_matrix))
            df = p * (p - 1) / 2
            p_value = chi2.sf(chi_square, df)
            
            print("\nBartlett's 球形檢定結果:")
            print("=" * 50)
            print(f"卡方值: {chi_square:.3f}")
            print(f"p-value: {p_value:.10f}")
            print(f"自由度: {int(df)}")
            
            if p_value < 0.05:
                print("結論: 拒絕虛無假設，數據適合進行因素分析")
            else:
                print("結論: 未能拒絕虛無假設，數據可能不適合進行因素分析")
                
            return chi_square, p_value
            
        except Exception as e:
            print(f"Bartlett 檢定過程中發生錯誤: {str(e)}")
            return None, None
        
    def calculate_sample_adequacy(self):
        """計算樣本適切性"""
        n_samples = self.X.shape[0]
        n_variables = self.X.shape[1]
        
        print("\n樣本適切性分析:")
        print("=" * 50)
        print(f"樣本數量: {n_samples}")
        print(f"變數數量: {n_variables}")
        print(f"樣本數/變數數比例: {n_samples/n_variables:.2f}")
        
        if n_samples/n_variables >= 20:
            adequacy = "理想"
        elif n_samples/n_variables >= 10:
            adequacy = "良好"
        elif n_samples/n_variables >= 5:
            adequacy = "尚可"
        else:
            adequacy = "不足"
            
        print(f"樣本適切性評價: {adequacy}")

def main():
    # 初始化分析器
    analyzer = PCATestAnalyzer("/Users/tommy/Desktop/應用多變量分析/processed_data_with_score.csv")
    
    # 準備數據
    analyzer.prepare_data()
    
    # 執行檢定
    analyzer.calculate_sample_adequacy()
    kmo_all, kmo_model = analyzer.perform_kmo_test()
    chi_square, p_value = analyzer.perform_bartlett_test()
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()