import pandas as pd
import numpy as np
import os

def load_csv(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded '{file_path}' successfully.\n")
        return df
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

def show_data_summary(df):
    print("\n🔍 FIRST 5 ROWS")
    print(df.head())

    print("\n📊 COLUMN TYPES")
    print(df.dtypes)

    print("\n🧼 MISSING VALUES")
    print(df.isnull().sum())

    print("\n📁 DUPLICATE ROWS")
    print(df.duplicated().sum())

def clean_missing_values(df):
    print("\n🧹 Cleaning missing values...")

    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype in ['float64', 'int64']:
                mean_val = df[column].mean()
                df[column] = df[column].fillna(mean_val)
                print(f"  ➤ Filled numeric '{column}' with mean: {mean_val:.2f}")
            else:
                mode_val = df[column].mode()[0] if not df[column].mode().empty else "UNKNOWN"
                df[column] = df[column].fillna(mode_val)
                print(f"  ➤ Filled text '{column}' with mode: {mode_val}")
    return df

def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"\n🗑️ Removed {before - after} duplicate rows.")
    return df

def remove_outliers_zscore(df, threshold=3):
    print("\n📉 Removing outliers using Z-score method...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    mask = (z_scores < threshold).all(axis=1)

    before = len(df)
    df_cleaned = df[mask]
    after = len(df_cleaned)

    print(f"  ➤ Removed {before - after} rows as outliers.")
    return df_cleaned

if __name__ == "__main__":
    file_path = "data/sample.csv"  # Put your CSV file here
    df = load_csv(file_path)

    if df is not None:
        show_data_summary(df)
        df = clean_missing_values(df)
        df = remove_duplicates(df)
        df = remove_outliers_zscore(df)
