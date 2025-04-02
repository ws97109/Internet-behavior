import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pc_scores_scatter(pc_scores, pc_x=1, pc_y=2):
    """
    繪製主成分得分散點圖
    
    Parameters:
    -----------
    pc_scores : DataFrame
        包含PC分數和分組資訊的DataFrame
    pc_x : int
        X軸要顯示的PC編號(1-4)
    pc_y : int
        Y軸要顯示的PC編號(1-4)
    """
    
    # 設置中文字體
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'Apple LiGothic Medium']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 創建圖形
    plt.figure(figsize=(12, 8))
    
    # 定義地區對應的標記
    markers = {
        '北部': 'o',  # 圓形
        '中部': 's',  # 方形
        '南部': '^',  # 三角形
        '東部': 'D',  # 菱形
        '其他': 'v'   # 倒三角
    }
    
    # 定義性別對應的顏色
    colors = {
        '男性': '#66B2FF',
        '女性': '#FF9999'
    }
    
    # 定義年齡組別對應的大小
    sizes = {
        '33-63': 100,
        '63-73': 150,
        '73-83': 200,
        '83-91': 250
    }
    
    # 繪製散點圖
    for gender in colors.keys():
        for region in markers.keys():
            #for age_group in sizes.keys():
                # 篩選數據
                mask = (pc_scores['gender_label'] == gender) & \
                       (pc_scores['region'] == region)
                data = pc_scores[mask]
                
                if len(data) > 0:
                    plt.scatter(data[f'PC{pc_x}'], 
                              data[f'PC{pc_y}'],
                              c=colors[gender],
                              marker=markers[region],
                              #s=sizes[age_group],
                              alpha=0.6,
                              label=f'{gender}-{region}')
    
    # 添加軸標籤
    plt.xlabel(f'PC{pc_x}')
    plt.ylabel(f'PC{pc_y}')
    plt.title(f'PC{pc_x} vs PC{pc_y} 主成分得分散點圖')
    
    # 添加圖例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加網格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 調整布局
    plt.tight_layout()
    
    return plt

def create_separate_legends(fig):
    """創建獨立的圖例說明"""
    # 創建新的圖形用於圖例
    legend_fig = plt.figure(figsize=(10, 2))
    
    # 性別圖例
    ax1 = legend_fig.add_subplot(131)
    ax1.scatter([], [], c='#66B2FF', label='男性')
    ax1.scatter([], [], c='#FF9999', label='女性')
    ax1.axis('off')
    ax1.legend(title='性別', loc='center')
    
    # 地區圖例
    ax2 = legend_fig.add_subplot(132)
    markers = {'北部': 'o', '中部': 's', '南部': '^', '東部': 'D', '其他': 'v'}
    for region, marker in markers.items():
        ax2.scatter([], [], c='gray', marker=marker, label=region)
    ax2.axis('off')
    ax2.legend(title='地區', loc='center')
    
    # 年齡組別圖例
    ax3 = legend_fig.add_subplot(133)
    sizes = {'33-63': 100, '63-73': 150, '73-83': 200, '83-91': 250}
    for age, size in sizes.items():
        ax3.scatter([], [], c='gray', s=size, label=age)
    ax3.axis('off')
    ax3.legend(title='年齡組別', loc='center')
    
    legend_fig.tight_layout()
    return legend_fig

def main():
    # 讀取數據
    df = pd.read_csv("/Users/tommy/Desktop/應用多變量分析/processed_data_with_score.csv")
    
    # 準備年齡組別數據
    bins = [33, 63, 73, 83, 91]
    labels = ['33-63', '63-73', '73-83', '83-91']
    df['age_group'] = pd.cut(df['q2'], bins=bins, labels=labels, include_lowest=True)
    
    # 準備地區數據
    region_map = {
        1: '北部', 2: '北部', 3: '北部', 4: '北部', 5: '北部',
        6: '北部',
        7: '中部', 8: '中部', 9: '中部', 10: '中部',
        11: '中部', 12: '中部', 13: '中部',
        14: '南部', 15: '南部', 16: '南部',
        17: '東部', 18: '東部', 19: '東部',
        20: '其他', 21: '其他', 22: '其他', 23: '其他', 24: '其他'
    }
    df['region'] = df['q3'].map(region_map)
    
    # 準備性別標籤
    df['gender_label'] = df['q1'].map({1.0: '男性', 2.0: '女性'})
    
    # 準備 PCA 數據
    attitude_groups = {
        'behavior_obs': [f'q22_0{i}_1' for i in range(1, 6)],
        'personal_act': [f'q23_0{i}_1' for i in range(1, 6)],
        'acceptance': [f'q25_0{i}_1' for i in range(1, 5)],
        'influence': [f'q26_0{i}_1' for i in range(1, 4)]
    }
    attitude_cols = [col for group in attitude_groups.values() for col in group]
    
    # 執行 PCA
    X = df[attitude_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)
    
    # 準備繪圖數據
    pc_scores = pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(4)]
    )
    pc_scores['age_group'] = df['age_group']
    pc_scores['gender_label'] = df['gender_label']
    pc_scores['region'] = df['region']
    
    # 繪製 PC1 vs PC2 散點圖
    scatter_plot = plot_pc_scores_scatter(pc_scores, pc_x=3, pc_y=2)
    scatter_plot.show()
    
    # 創建獨立的圖例說明
    legend_plot = create_separate_legends(scatter_plot)
    legend_plot.show()

if __name__ == "__main__":
    main()