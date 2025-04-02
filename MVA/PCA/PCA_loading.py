import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle

# 設置中文字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'Apple LiGothic Medium']
plt.rcParams['axes.unicode_minus'] = False

class PCAAnalyzer:
    def __init__(self, data_path):
        """初始化 PCA 分析器"""
        self.df = pd.read_csv(data_path)
        self.X = None
        self.attitude_cols = None
        self.attitude_groups = None
        self.pca = None
        self.X_pca = None
        self.X_scaled = None
        self.loadings = None
        
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
        
    def do_pca(self):
        """執行 PCA 分析"""
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        
        # 計算最佳主成分數
        pca_full = PCA()
        X_pca_full = pca_full.fit_transform(self.X_scaled)
        
        # 使用 Kaiser 準則和解釋變異量比例確定主成分數
        n_components = sum(pca_full.explained_variance_ > 1)
        var_ratio_cum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components_var = np.argmax(var_ratio_cum > 0.8) + 1
        
        # 選擇較小的數量
        n_components = min(n_components, n_components_var)
        
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        # 計算 loadings
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.attitude_cols
        )
        
    def plot_scree(self):
        """繪製改進的碎石圖與累積解釋變異量圖"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        variance_ratio = self.pca.explained_variance_ratio_
        cumulative_ratio = np.cumsum(variance_ratio)
        
        # 碎石圖
        ax1.plot(range(1, len(variance_ratio) + 1), variance_ratio, 'bo-')
        ax1.plot(range(1, len(variance_ratio) + 1), variance_ratio, 'r--', alpha=0.5)
        ax1.set_title('碎石圖與Kaiser準則', fontsize=12)
        ax1.axhline(y=1/len(variance_ratio), color='g', linestyle='--', 
                    label='Average criterion (1/p)')
        ax1.set_xlabel('主成分數')
        ax1.set_ylabel('解釋變異量')
        ax1.legend()
        
        # 累積解釋變異量圖
        ax2.plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio, 'ro-')
        ax2.axhline(y=0.8, color='g', linestyle='--', label='80% threshold')
        ax2.set_title('累積解釋變異量', fontsize=12)
        ax2.set_xlabel('主成分數')
        ax2.set_ylabel('累積解釋變異量比例')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_loadings_heatmap(self):
        """繪製主成分負荷量熱力圖"""
        plt.figure(figsize=(15, 10))
        loadings_display = self.loadings.iloc[:, :4]  # 只顯示前4個主成分
        
        sns.heatmap(loadings_display, annot=True, cmap='coolwarm', center=0,
                    fmt='.3f', annot_kws={'size': 8})
        plt.title('主成分負荷量熱力圖 (前4個主成分)', fontsize=12, pad=20)
        plt.xlabel('主成分', fontsize=10)
        plt.ylabel('變數', fontsize=10)
        plt.tight_layout()
        plt.show()
        
    def analyze_components(self):
        """分析主成分結果"""
        n_components = min(4, self.pca.n_components_)
        print("\n主成分分析結果:")
        print("=" * 50)
        
        total_var = np.sum(self.pca.explained_variance_ratio_[:n_components])
        print(f"\n前{n_components}個主成分總解釋變異量: {total_var:.2%}")
        
        for i in range(n_components):
            pc = f'PC{i + 1}'
            print(f"\n{pc} (解釋變異量 {self.pca.explained_variance_ratio_[i]:.2%})")
            print("-" * 40)
            
            component_loadings = self.loadings[pc].sort_values(ascending=False)
            print("最重要的正向負荷量:")
            print(component_loadings[component_loadings > 0.3][:3])
            print("\n最重要的負向負荷量:")
            print(component_loadings[component_loadings < -0.3][:3])
        
        print("\n變異量解釋表:")
        print("=" * 50)
        variance_table = pd.DataFrame({
            '特徵值': self.pca.explained_variance_[:n_components],
            '變異量比例': self.pca.explained_variance_ratio_[:n_components],
            '累積變異量': np.cumsum(self.pca.explained_variance_ratio_[:n_components])
        })
        print(variance_table.round(4))

def main():
    # 初始化分析器
    analyzer = PCAAnalyzer("/Users/tommy/Desktop/應用多變量分析/processed_data_with_score.csv")
    
    # 執行分析
    analyzer.prepare_data()
    analyzer.do_pca()
    
    # 生成視覺化
    analyzer.plot_scree()
    analyzer.plot_loadings_heatmap()
    
    # 分析結果
    analyzer.analyze_components()
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()