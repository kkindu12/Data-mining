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

#Vizualization of outliers
# Step 2a: Draw boxplot to visually highlight outliers
print("\n" + "="*50)
print("STEP 2a: VISUALIZING OUTLIERS WITH BOXPLOT")
print("="*50)

plt.figure(figsize=(12, 6))

# Create boxplot for fare column (chosen for visualization)
plt.subplot(1, 2, 1)
boxplot = plt.boxplot(data['fare'], patch_artist=True)
plt.title('Boxplot of Fare Column', fontsize=14, fontweight='bold')
plt.ylabel('Fare ($)')

# Customize boxplot colors
boxplot['boxes'][0].set_facecolor('lightblue')
boxplot['medians'][0].set_color('red')
boxplot['fliers'][0].set_marker('o')
boxplot['fliers'][0].set_markerfacecolor('red')
boxplot['fliers'][0].set_markeredgecolor('red')
boxplot['fliers'][0].set_alpha(0.6)

# Add value annotations
fare_stats = data['fare'].describe()
plt.text(1.2, fare_stats['75%'], f"Q3: {fare_stats['75%']:.1f}", va='center')
plt.text(1.2, fare_stats['25%'], f"Q1: {fare_stats['25%']:.1f}", va='center')
plt.text(1.2, fare_stats['50%'], f"Median: {fare_stats['50%']:.1f}", va='center')

plt.grid(True, alpha=0.3)

# Step 2a: Create scatter plot to highlight outliers
print("Creating scatter plot visualization...")

plt.subplot(1, 2, 2)

# Detect outliers in fare column for highlighting
fare_outliers, lower_bound, upper_bound = detect_outliers_iqr(data['fare'])

# Create a mask for outliers
is_outlier = data['fare'] > upper_bound  # Only upper outliers for fare

# Plot normal points
plt.scatter(data[~is_outlier]['age'], data[~is_outlier]['fare'], 
           alpha=0.6, color='blue', s=50, label='Normal Data')

# Plot outliers with different style
plt.scatter(data[is_outlier]['age'], data[is_outlier]['fare'], 
           alpha=0.8, color='red', s=80, marker='X', label='Outliers', 
           edgecolors='darkred', linewidth=1)

plt.xlabel('Age (years)')
plt.ylabel('Fare ($)')
plt.title('Age vs Fare - Outliers Highlighted', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outlier_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualization saved as 'outlier_visualization.png'")
 
# Step 2b: Write a short observation
print("\n" + "="*50)
print("STEP 2b: VISUALIZATION OBSERVATION")
print("="*50)

fare_outliers_count = summary_data[1]['IQR_Outliers']  # Fare is second column

observation = f"""{fare_outliers_count} extreme points were identified as outliers in the Fare column. 
These represent passengers who paid exceptionally high ticket prices compared to the majority, 
with some fares exceeding ${upper_bound:.0f} while most passengers paid less than ${data['fare'].quantile(0.75):.0f}."""

print("📝 OBSERVATION:")
print(observation)

# Step 3a: Apply outlier treatment using capping/winsorizing
print("\n" + "="*50)
print("STEP 3a: HANDLING OUTLIERS - CAPPING METHOD")
print("="*50)

def cap_outliers(column):
    """
    Cap outliers using IQR method (Winsorizing)
    Values below lower bound are set to lower bound
    Values above upper bound are set to upper bound
    """
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"  • Original range: [{column.min():.2f}, {column.max():.2f}]")
    print(f"  • Capping bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Cap the values
    capped_column = column.clip(lower=lower_bound, upper=upper_bound)
    
    return capped_column

print("Applying capping to Fare column...")
data_cleaned = data.copy()
data_cleaned['fare_capped'] = cap_outliers(data['fare'])

print("✅ Outlier treatment completed!")