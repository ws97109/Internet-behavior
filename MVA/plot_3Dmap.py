import plotly.graph_objects as go
import geopandas as gpd
import numpy as np
from plotly.subplots import make_subplots

# 讀取台灣地圖 shapefile
taiwan_map = gpd.read_file('taiwan_map/COUNTY_MOI_1130718.shp')

# 確保座標系統為 WGS84
taiwan_map = taiwan_map.to_crs('EPSG:4326')

# 定義台灣各區域的中心點座標
regions = ['北部', '中部', '南部', '東部']
region_coords = {
    '北部': {'lat': 25.0374865, 'lon': 121.5637262},
    '中部': {'lat': 24.1477358, 'lon': 120.6736482},
    '南部': {'lat': 22.9998999, 'lon': 120.2268758},
    '東部': {'lat': 23.9871589, 'lon': 121.6015714}
}

# 準備資料 - 將數值縮小為原來的1/3
gender_data = {
    '北部': {'男': 64.7/3, '女': 35.3/3},
    '中部': {'男': 55.9/3, '女': 44.1/3},
    '南部': {'男': 61.4/3, '女': 38.6/3},
    '東部': {'男': 68.6/3, '女': 31.4/3}
}

internet_usage = {
    '北部': {'0-3h': 41.1/3, '3-6h': 40.7/3, '6h+': 18.2/3},
    '中部': {'0-3h': 45.3/3, '3-6h': 38.4/3, '6h+': 16.3/3},
    '南部': {'0-3h': 39.2/3, '3-6h': 42.1/3, '6h+': 18.7/3},
    '東部': {'0-3h': 54.3/3, '3-6h': 34.3/3, '6h+': 11.4/3}
}

# 創建主圖表
fig = go.Figure()

# 添加台灣地圖底圖
for idx, row in taiwan_map.iterrows():
    if row.geometry.geom_type == 'MultiPolygon':
        for geom in row.geometry.geoms:
            x, y = geom.exterior.xy
            fig.add_trace(go.Scatter3d(
                x=list(x),
                y=list(y),
                z=[0]*len(x),
                mode='lines',
                line=dict(color='gray', width=2),  # 增加線條寬度
                showlegend=False
            ))
    else:
        x, y = row.geometry.exterior.xy
        fig.add_trace(go.Scatter3d(
            x=list(x),
            y=list(y),
            z=[0]*len(x),
            mode='lines',
            line=dict(color='gray', width=2),  # 增加線條寬度
            showlegend=False
        ))

# 為每個區域添加數據柱
for region in regions:
    # 性別分布
    fig.add_trace(go.Scatter3d(
        x=[region_coords[region]['lon'], region_coords[region]['lon']],
        y=[region_coords[region]['lat'], region_coords[region]['lat']],
        z=[0, gender_data[region]['男']],
        mode='lines',
        line=dict(color='blue', width=8),  # 增加柱狀圖寬度
        name=f'{region}-男性比例'
    ))
    
    # 網路使用時間
    z_values = [internet_usage[region]['0-3h'],
                internet_usage[region]['3-6h'],
                internet_usage[region]['6h+']]
    
    for i, z in enumerate(z_values):
        fig.add_trace(go.Scatter3d(
            x=[region_coords[region]['lon'] + 0.15],  # 增加間距
            y=[region_coords[region]['lat'] + 0.15 * i],
            z=[0, z],
            mode='lines',
            line=dict(
                color=['lightgreen', 'green', 'darkgreen'][i],
                width=8  # 增加柱狀圖寬度
            ),
            name=f'{region}-網路使用{["0-3h", "3-6h", "6h+"][i]}'
        ))

# 更新布局
fig.update_layout(
    title='台灣各區域人口特徵與網路使用分析',
    scene = dict(
        xaxis_title='經度',
        yaxis_title='緯度',
        zaxis_title='百分比',
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.5, y=0.5, z=1.5)  # 調整視角使地圖看起來更大
        ),
        aspectmode='cube',  # 改用立方體模式以調整比例
        aspectratio=dict(x=2, y=2, z=1)  # 調整xyz軸的比例
    ),
    height=1000,  # 增加圖表高度
    width=1200,   # 增加圖表寬度
    showlegend=True
)

# 添加註解說明
fig.add_annotation(
    text='藍色: 男性比例 | 綠色漸層: 網路使用時間分布',
    xref='paper', yref='paper',
    x=0, y=1.1,
    showarrow=False,
    font=dict(size=14)  # 增加字體大小
)

fig.show()