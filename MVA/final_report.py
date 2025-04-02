import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 設置字體大小
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

# 讀取CSV檔案到DataFrame
df = pd.read_csv('/Users/lishengfeng/Desktop/多變量分析/newselect_onehot(1).csv')

# 替換欄位值
df['q1'] = df['q1'].replace({0: 'female', 1: 'man'})
df['q3'] = df['q3'].replace({1: 'North', 2: 'Central', 3: 'South', 4: 'East', 5: 'Islands', 6: 'Others'})
df['q7'] = df['q7'].replace({1: '0-3 hrs', 2: '3-6 hrs', 3: 'over 6 hrs'})

def categorize_birth_year(year):
    if year <= 60:
        return 'Before 60'
    elif 61 <= year <= 70:
        return '61-70'
    elif 71 <= year <= 80:
        return '71-80'
    elif 81 <= year <= 90:
        return '81-90'
    else:
        return 'After 90'

df['Birth_Category'] = df['q2'].apply(categorize_birth_year)

# 重命名欄位
df.rename(columns={
    'q1': 'Gender',
    'q3': 'Area',
    'q2': 'Birth_Year',
    'q7': 'Net_Time'
}, inplace=True)

# 定義調色盤
light_to_dark_palette = ['#FFC0CB', '#FF99CC', '#FF69B4', '#FF1493', '#DB7093', '#C71585', '#8B0000']

# 創建性別分布圓餅圖
plt.figure(figsize=(8, 8))
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=light_to_dark_palette[:2])
plt.title('Gender Distribution')
plt.axis('equal')
plt.legend(gender_counts.index, title="Gender", bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.show()

# 創建出生年份區間分布圓餅圖
plt.figure(figsize=(8, 8))
birth_counts = df['Birth_Category'].value_counts().reindex(['Before 60', '61-70', '71-80', '81-90', 'After 90'])
plt.pie(birth_counts, labels=birth_counts.index, autopct='%1.1f%%', startangle=140, colors=light_to_dark_palette[:5])
plt.title('Birth Year Distribution')
plt.axis('equal')
plt.legend(birth_counts.index, title="Birth Year Range\n(Minguo Calendar)", bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.show()

# 創建地區分布圓餅圖
plt.figure(figsize=(8, 8))
area_counts = df['Area'].value_counts()
plt.pie(area_counts, labels=area_counts.index, autopct='%1.1f%%', startangle=140, colors=light_to_dark_palette)
plt.title('Area Distribution')
plt.axis('equal')
plt.legend(area_counts.index, title="Area", bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.show()

# 創建網路使用時間分布圓餅圖
plt.figure(figsize=(8, 8))
net_time_counts = df['Net_Time'].value_counts()
plt.pie(net_time_counts, labels=net_time_counts.index, autopct='%1.1f%%', startangle=140, colors=light_to_dark_palette[:3])
plt.title('Net Time Distribution')
plt.axis('equal')
plt.legend(net_time_counts.index, title="Net Time", bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_gender_distribution():
    plt.figure(figsize=(12, 6))
    
    # 計算百分比
    gender_area = pd.crosstab(df['Area'], df['Gender'], normalize='index') * 100
    
    # 繪製堆疊條形圖
    ax = gender_area.plot(kind='bar', stacked=True, color=['#FFC0CB', '#FF69B4'])
    
    # 添加百分比標籤
    for i in range(len(gender_area.index)):
        female_pct = gender_area.iloc[i, 0]
        male_pct = gender_area.iloc[i, 1]
        
        ax.text(i, female_pct/2, f'{female_pct:.1f}%', ha='center', va='center')
        ax.text(i, female_pct + male_pct/2, f'{male_pct:.1f}%', ha='center', va='center')
    
    plt.title('Gender Distribution by Area')
    plt.xlabel('Area')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Gender', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_birth_distribution():
    plt.figure(figsize=(12, 6))
    
    # 計算百分比
    birth_area = pd.crosstab(df['Area'], df['Birth_Category'], normalize='index') * 100
    birth_area = birth_area[['Before 60', '61-70', '71-80', '81-90', 'After 90']]
    
    # 繪製堆疊條形圖
    ax = birth_area.plot(kind='bar', stacked=True,
                    color=['#FFC0CB', '#FFB6C1', '#FF69B4', '#FF1493', '#C71585'])
    
    # 添加百分比標籤
    yoff = 0
    for i in range(len(birth_area.index)):
        yoff = 0
        for j in range(len(birth_area.columns)):
            value = birth_area.iloc[i, j]
            ax.text(i, yoff + value/2, f'{value:.1f}%', ha='center', va='center')
            yoff += value
    
    plt.title('Birth Year Distribution by Area')
    plt.xlabel('Area')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Birth Year Range', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_nettime_distribution():
    plt.figure(figsize=(12, 6))
    
    # 計算百分比
    net_area = pd.crosstab(df['Area'], df['Net_Time'], normalize='index') * 100
    
    # 繪製堆疊條形圖
    ax = net_area.plot(kind='bar', stacked=True, 
                  color=['#FFC0CB', '#FF69B4', '#C71585'])
    
    # 添加百分比標籤
    yoff = 0
    for i in range(len(net_area.index)):
        yoff = 0
        for j in range(len(net_area.columns)):
            value = net_area.iloc[i, j]
            ax.text(i, yoff + value/2, f'{value:.1f}%', ha='center', va='center')
            yoff += value
    
    plt.title('Internet Usage Time Distribution by Area')
    plt.xlabel('Area')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Net Time', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()

# 依序執行三個圖表
plot_gender_distribution()
plot_birth_distribution()
plot_nettime_distribution()