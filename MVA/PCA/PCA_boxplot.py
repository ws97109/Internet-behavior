import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 設置中文字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'Apple LiGothic Medium']
plt.rcParams['axes.unicode_minus'] = False

def plot_pc_scores_unified(pc_scores, by='age'):
    """繪製主成分得分分布圖，統一Y軸尺度"""
    
    if by == 'age':
        x_col = 'age_group'
        x_label = '出生民國年組別'
    else:  # by == 'region'
        x_col = 'region'
        x_label = '地區'
    
    # 找出所有PC分數的最大和最小值，用於統一Y軸尺度
    y_min = min([pc_scores[f'PC{i+1}'].min() for i in range(4)])
    y_max = max([pc_scores[f'PC{i+1}'].max() for i in range(4)])
    
    # 為了留些邊距，稍微擴大範圍
    y_range = [y_min - 0.5, y_max + 0.5]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    # 定義顏色方案
    colors = {'男性': '#66B2FF', '女性': '#FF9999'}
    
    for i in range(4):
        sns.boxplot(x=x_col, y=f'PC{i+1}', hue='gender_label',
                   data=pc_scores, ax=axes[i],
                   palette=colors)
        axes[i].set_title(f'PC{i+1} 得分分布')
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel('主成分得分')
        axes[i].set_ylim(y_range)  # 統一Y軸範圍
        
        # 調整圖例
        axes[i].legend(title='性別')
    
    plt.tight_layout()
    plt.show()

# 主程式
def main():
    # 讀取數據
    df = pd.read_csv("/Users/tommy/Desktop/應用多變量分析/processed_data_with_score.csv")
    
    # 準備年齡組別數據
    bins = [33, 63, 73, 83, 91]
    labels = ['33-63', '63-73', '73-83', '83-91']
    df['age_group'] = pd.cut(df['q2'], bins=bins, labels=labels, include_lowest=True)
    
    # 準備地區數據
    region_map = {
        1: '北部', 2: '北部', 3: '北部', 4: '北部', 5: '北部',  # 基隆、台北、新北、桃園、新竹縣
        6: '北部',  # 新竹市
        7: '中部', 8: '中部', 9: '中部', 10: '中部',  # 苗栗、南投、台中、彰化
        11: '中部', 12: '中部', 13: '中部',  # 雲林、嘉義縣、嘉義市
        14: '南部', 15: '南部', 16: '南部',  # 台南、高雄、屏東
        17: '東部', 18: '東部', 19: '東部',  # 宜蘭、花蓮、台東
        20: '其他', 21: '其他', 22: '其他', 23: '其他', 24: '其他'  # 澎湖、金門、連江、外島
    }
    df['region'] = df['q3'].map(region_map)
    
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
    pc_scores['gender_label'] = df['q1'].map({1.0: '男性', 2.0: '女性'})
    pc_scores['region'] = df['region']
    
    # 繪製圖表
    print("依年齡組別分析：")
    plot_pc_scores_unified(pc_scores, by='age')
    print("\n依地區分析：")
    plot_pc_scores_unified(pc_scores, by='region')

if __name__ == "__main__":
    main()