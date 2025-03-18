import pandas as pd
import numpy as np

def feature_engineering(df):
    """
    Perform feature engineering on the mobile phone dataset.
    Returns a processed DataFrame with new features.
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # --- Step 0: Data Transformation ---
    # Change Price to USD
    df_processed["Price (USD)"] = round(df_processed.Price*0.011).astype(int)
    df_processed.drop(["Price"], axis=1, inplace=True)
    
    # Change RAM to GB
    df_processed["RAM (GB)"] = round(df_processed["RAM (MB)"]/1000, 2)
    df_processed.drop(["RAM (MB)"], axis=1, inplace=True)
    
    # Binary Variables
    binary_variable_list = list()
    
    for x in df_processed.columns:
        if df_processed[x].value_counts().index.to_list() == ['Yes', 'No']:
            binary_variable_list.append(x)
    
    for x in binary_variable_list:
        df_processed[x] = df_processed[x].map({"Yes": 1, "No": 0})
    
    # Pixel
    df_processed["Pixel per inch (PPI)"] = np.sqrt((df_processed["Resolution x"]**2 + df_processed["Resolution y"]**2))/df_processed["Screen size (inches)"]
    df_processed["Pixel per inch (PPI)"] = df_processed["Pixel per inch (PPI)"].round(2)
    
    # Brand help
    brands_by_country = {
        "USA": ["Apple", "Google", "HP", "Microsoft", "Razer", "Cat", "Blu", "BlackBerry", "Motorola", "Nuu Mobile"],
        "South Korea": ["Samsung", "LG"],
        "China": ["10.or", "Black Shark", "Coolpad", "Gionee", "Honor", "Huawei", "Lenovo", "Meizu", "Nubia", "OnePlus", "Oppo", "Realme", "Vivo", "Xiaomi", "ZTE", "Zopo", "Phicomm", "Zuk", "LeEco", "Homtom", "Poco", "Sansui", "TCL"],
        "Taiwan": ["Acer", "Asus", "HTC"],
        "Japan": ["Sony", "Panasonic", "Sharp"],
        "India": ["Aqua", "Billion", "Celkon", "Comio", "InFocus", "Intex", "Itel", "Jio", "Jivi", "Karbonn", "Kult", "Lava", "Lephone", "Lyf", "M-tech", "Micromax", "Mobiistar", "Onida", "Reach", "Smartron", "Spice", "Swipe", "Tambo", "Videocon", "Xolo", "Yu", "Zen", "Ziox", "mPhone", "iBall", "iVoomi"],
        "EU": ["Nokia", "Alcatel","Gigaset", "Philips"],
        "Hong Kong": ["Infinix", "Tecno", "Itel"],
    }
    
    # Brand top price
    top_price = dict()
    for x in df_processed["Brand"].unique():
        top_price[x] = df_processed[df_processed["Brand"] == x]["Price (USD)"].max()
    
    df_processed["Brand Top Price"] = df_processed["Brand"].map(top_price)
    
    # Reverse the keys and values in the dictionary
    brand_to_country = {}
    for country, brands in brands_by_country.items():
        for brand in brands:
            brand_to_country[brand] = country
    
    # Map them to the dataset
    df_processed["Brand Origin"] = df_processed["Brand"].map(brand_to_country)
    
    # Setting price ranges
    price_range_label = ["Ultra Budget", "Budget", "Mid Range", "Upper Mid", "Premium", "Flagship"]
    price_range_bins = [0, 100, 250, 400, 700, 1000, np.inf]
    df_processed["Price Range"] = pd.cut(df_processed["Price (USD)"], bins=price_range_bins, labels=price_range_label)
    
    # --- Step 1: Encode Operating System ---
    def categorize_os(os):
        os = os.lower()
        if 'android' in os:
            return 'Android'
        elif 'windows' in os:
            return 'Windows'
        elif 'ios' in os:
            return 'iOS'
        else:
            return 'Other'
    
    df_processed['Operating system'] = df_processed['Operating system'].apply(categorize_os)
    
    # --- Step 2: Convert Price Range into Ordinal Encoding ---
    price_range_mapping = {
        "Ultra Budget": 0, 
        "Budget": 1, 
        "Mid Range": 2, 
        "Upper Mid": 3, 
        "Premium": 4, 
        "Flagship": 5
    }
    df_processed['Price Range'] = df_processed['Price Range'].map(price_range_mapping)
    
    # --- Step 3: Create Interaction Terms ---
    df_processed['Camera Score'] = df_processed['Rear camera'] + df_processed['Front camera']
    df_processed['Performance Score'] = (df_processed['RAM (GB)'] * 2) + (df_processed['Internal storage (GB)'] / 64)
    df_processed['Battery-to-Screen Ratio'] = df_processed['Battery capacity (mAh)'] / df_processed['Screen size (inches)']
    
    return df_processed