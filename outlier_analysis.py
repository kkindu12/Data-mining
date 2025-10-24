# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

print("=== PART B: DETECTING OUTLIERS ===\n")

# load the titanic dataset
df = sns.load_dataset('titanic')

# Step 1a: Select two numeric columns
print("Step 1a: Selecting two numeric columns...")
selected_columns = ['age', 'fare']
data = df[selected_columns].copy()

print(f"Selected columns: {selected_columns}")
print(f"Dataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())

# Handle missing values for proper analysis
print("\nStep 1a (cont.): Handling missing values...")
print("Missing values before handling:")
print(data.isnull().sum())

# Fill missing age values with median
data['age'].fillna(data['age'].median(), inplace=True)

print("\nMissing values after handling:")
print(data.isnull().sum())

# Step 1b: IQR Method for outlier detection
print("\n" + "="*50)
print("STEP 1b: IQR METHOD OUTLIER DETECTION")
print("="*50)

def detect_outliers_iqr(column):
    """
    Detect outliers using Interquartile Range (IQR) method
    Outliers: values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
    """
    Q1 = column.quantile(0.25)  # 25th percentile
    Q3 = column.quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1               # Interquartile Range
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    
    return outliers, lower_bound, upper_bound

# Detect outliers for both columns using IQR
print("\nIQR Method Results:")
print("-" * 30)

iqr_results = {}
for col in selected_columns:
    outliers, lower_bound, upper_bound = detect_outliers_iqr(data[col])
    iqr_results[col] = len(outliers)
    
    print(f"\n{col.upper()} Column:")
    print(f"  • Lower bound: {lower_bound:.2f}")
    print(f"  • Upper bound: {upper_bound:.2f}")
    print(f"  • Outliers detected: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  • Outlier values: {outliers.values[:5]}")  

    # Step 1b: Z-Score Method for outlier detection
print("\n" + "="*50)
print("STEP 1b: Z-SCORE METHOD OUTLIER DETECTION")
print("="*50)

def detect_outliers_zscore(column, threshold=3):
    """
    Detect outliers using Z-score method
    Outliers: values with |Z-score| > threshold (default: 3)
    """
    z_scores = np.abs(stats.zscore(column))
    outliers = column[z_scores > threshold]
    
    return outliers, z_scores

# Detect outliers for both columns using Z-score
print("\nZ-Score Method Results:")
print("-" * 30)

zscore_results = {}
for col in selected_columns:
    outliers, z_scores = detect_outliers_zscore(data[col])
    zscore_results[col] = len(outliers)
    
    print(f"\n{col.upper()} Column:")
    print(f"  • Z-score range: {z_scores.min():.2f} to {z_scores.max():.2f}")
    print(f"  • Outliers detected: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  • Max Z-score: {z_scores.max():.2f}")
    
# Step 1c: Report how many outliers were detected by each method
print("\n" + "="*50)
print("STEP 1c: OUTLIER DETECTION SUMMARY REPORT")
print("="*50)

# Create summary table
summary_data = []
for col in selected_columns:
    iqr_outliers, _, _ = detect_outliers_iqr(data[col])
    z_outliers, _ = detect_outliers_zscore(data[col])
    
    summary_data.append({
        'Column': col,
        'IQR_Outliers': len(iqr_outliers),
        'Zscore_Outliers': len(z_outliers),
        'Total_Values': len(data[col]),
        'IQR_Percentage': (len(iqr_outliers) / len(data[col]) * 100),
        'Zscore_Percentage': (len(z_outliers) / len(data[col]) * 100)
    })

# Create and display summary dataframe
summary_df = pd.DataFrame(summary_data)
print("\n📊 OUTLIER DETECTION RESULTS:")
print(summary_df.to_string(index=False))

print("\n🔍 KEY FINDINGS:")
for result in summary_data:
    print(f"• {result['Column'].upper()}: {result['IQR_Outliers']} outliers (IQR), {result['Zscore_Outliers']} outliers (Z-score)")    