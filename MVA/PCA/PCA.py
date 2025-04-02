import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 設定中文字體
plt.rcParams['font.family'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_output_directory(directory_name='output_figures'):
    """建立輸出圖片的目錄"""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    return directory_name

def load_and_prepare_data(file_path):
    """讀取和準備資料"""
    # 讀取資料
    df = pd.read_csv(file_path)
    print(f"資料維度：{df.shape}")
    
    # 顯示基本統計資訊
    print("\n資料基本統計：")
    print(df.describe())
    
    return df

def preprocess_data_for_pca(df):
    """資料預處理"""
    # 1. 檢查缺失值
    print("\n檢查缺失值：")
    print(df.isnull().sum())
    
    # 2. 處理缺失值
    # 對於社群媒體和影音平台的使用情況，將缺失值填充為0（表示不使用）
    social_media_cols = [col for col in df.columns if any(prefix in col for prefix in ['社群_', '即時通訊_', '影音_'])]
    df[social_media_cols] = df[social_media_cols].fillna(0)
    
    # 對於其他數值變數，使用中位數填充
    numeric_cols = ['網路行為規範', '霸凌行為', '負面影響認知', '衝突容忍度', '上網時間']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 對於類別變數，使用眾數填充
    categorical_cols = ['性別', '職業', '教育程度']
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    # 再次檢查是否還有缺失值
    if df.isnull().sum().any():
        print("\n警告：資料中仍存在缺失值")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # 標準化資料
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # 轉換為DataFrame以保留變數名稱
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    
    return scaled_df, scaler

def perform_pca(scaled_data):
    """執行PCA分析"""
    # 初始化PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # 計算解釋變異量
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # 輸出結果
    print("\nPCA分析結果：")
    for i, (var_ratio, cum_ratio) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio), 1):
        print(f"主成分{i}:")
        print(f"- 解釋變異量: {var_ratio:.4f}")
        print(f"- 累積解釋變異量: {cum_ratio:.4f}")
    
    return pca, pca_result

def plot_scree(pca, output_dir):
    """繪製碎石圖"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, 'bo-')
    plt.title('碎石圖')
    plt.xlabel('主成分數')
    plt.ylabel('解釋變異量比例')
    plt.grid(True)
    
    # 保存圖片
    plt.savefig(os.path.join(output_dir, 'scree_plot_pca.png'))
    plt.close()

def plot_cumulative_variance(pca, output_dir):
    """繪製累積解釋變異量圖"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
            np.cumsum(pca.explained_variance_ratio_), 'ro-')
    plt.axhline(y=0.8, color='k', linestyle='--')
    plt.title('累積解釋變異量')
    plt.xlabel('主成分數')
    plt.ylabel('累積解釋變異量比例')
    plt.grid(True)
    
    # 保存圖片
    plt.savefig(os.path.join(output_dir, 'cumulative_variance_pca.png'))
    plt.close()

def plot_component_loading(pca, feature_names, n_components, output_dir):
    """繪製成分負荷量熱圖"""
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
    plt.title('主成分負荷量')
    
    # 保存圖片
    plt.savefig(os.path.join(output_dir, 'component_loadings_pca.png'))
    plt.close()
    
    return loadings

def plot_biplot(pca_result, loadings, feature_names, output_dir):
    """繪製雙標圖"""
    plt.figure(figsize=(12, 8))
    
    # 繪製散點圖
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    
    # 繪製特徵向量
    for i, (x, y) in enumerate(zip(loadings['PC1'], loadings['PC2'])):
        plt.arrow(0, 0, x*5, y*5, color='r', alpha=0.5)
        plt.text(x*5.2, y*5.2, feature_names[i], color='r')
    
    plt.title('PCA雙標圖')
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    
    # 保存圖片
    plt.savefig(os.path.join(output_dir, 'biplot_pca.png'))
    plt.close()

def main():
    try:
        print("開始執行PCA分析...")
        output_dir = create_output_directory()
        
        # 讀取資料
        df = load_and_prepare_data('./output_figures/combined_data_for_analysis.csv')
        
        # 檢查數據品質
        print("\n數據品質檢查：")
        print(f"原始資料維度：{df.shape}")
        print(f"缺失值數量：\n{df.isnull().sum()}")

        # 資料預處理
        scaled_df, scaler = preprocess_data_for_pca(df)
        
        # 確認預處理後沒有缺失值
        if scaled_df.isnull().sum().any():
            raise ValueError("預處理後資料仍包含缺失值")
        
        # 執行PCA
        pca, pca_result = perform_pca(scaled_df)
        
        # 繪製視覺化圖表
        plot_scree(pca, output_dir)
        plot_cumulative_variance(pca, output_dir)
        
        # 根據碎石圖選擇合適的主成分數量
        n_components = 4  # 可以根據實際結果調整
        
        # 繪製成分負荷量圖
        loadings = plot_component_loading(pca, df.columns, n_components, output_dir)
        
        # 繪製雙標圖
        plot_biplot(pca_result, loadings, df.columns, output_dir)
        
        # 儲存PCA結果
        pca_df = pd.DataFrame(
            pca_result[:, :n_components],
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        pca_df.to_csv(os.path.join(output_dir, 'pca_results.csv'), index=False)
        
        print("PCA分析完成，結果已儲存至output_figures資料夾")
        
        return pca, pca_result, loadings
        
    except Exception as e:
        print(f"執行過程中發生錯誤：{str(e)}")
        return None, None, None

if __name__ == "__main__":
    main()