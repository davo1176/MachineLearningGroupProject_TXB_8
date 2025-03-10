import pandas as pd
import numpy as np

def feature_engineering(df):

    df_processed = df.copy()
    
    price_ranges = [0, 5000, 10000, 20000, float('inf')]
    labels = ['budget', 'low_mid', 'high_mid', 'premium']
    df_processed['price_range'] = pd.cut(df_processed['Price'], bins=price_ranges, labels=labels, right=False)
    
    df_processed['screen_area'] = df_processed['Screen size (inches)'] ** 2
    df_processed['resolution_total'] = df_processed['Resolution x'] * df_processed['Resolution y']
    df_processed['pixel_density'] = np.sqrt(df_processed['resolution_total']) / df_processed['Screen size (inches)']
    
    df_processed['RAM_GB'] = df_processed['RAM (MB)'] / 1000
    
    df_processed['performance_score'] = df_processed['Processor'] * df_processed['RAM_GB']
    
    df_processed['camera_score'] = df_processed['Rear camera'] + 0.5 * df_processed['Front camera']
    
    binary_features = ['Touchscreen', 'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    for feature in binary_features:
        df_processed[feature] = df_processed[feature].map({'Yes': 1, 'No': 0})
    
    connectivity_features = ['Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    df_processed['connectivity_score'] = df_processed[connectivity_features].sum(axis=1)
    
    brand_avg_price = df.groupby('Brand')['Price'].mean().reset_index()
    brand_avg_price.columns = ['Brand', 'brand_avg_price']
    df_processed = pd.merge(df_processed, brand_avg_price, on='Brand', how='left')
    
    df_processed['storage_ram_ratio'] = df_processed['Internal storage (GB)'] / df_processed['RAM_GB']

    df_processed['OS_category'] = df_processed['Operating system'].map(
        lambda x: 'iOS' if x == 'iOS' else ('Android' if x == 'Android' else 'Other')
    )
    
    drop_cols = ['Unnamed: 0', 'Name', 'Model', 'Resolution x', 'Resolution y', 
                'RAM (MB)', 'Price', 'Operating system'] 
    
    drop_cols = [col for col in drop_cols if col in df_processed.columns]
    df_processed = df_processed.drop(columns=drop_cols)

    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0)
    
    cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col != 'price_range':  # Don't modify the target variable
            most_frequent = df_processed[col].mode()[0]
            df_processed[col] = df_processed[col].fillna(most_frequent)
    
    if 'storage_ram_ratio' in df_processed.columns:
        df_processed['storage_ram_ratio'] = df_processed['storage_ram_ratio'].replace([np.inf, -np.inf], 0)
    
    return df_processed