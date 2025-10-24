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

# Step 3b: Compare mean and standard deviation before and after cleaning
print("\n" + "="*50)
print("STEP 3b: COMPARISON - BEFORE vs AFTER OUTLIER TREATMENT")
print("="*50)

# Calculate statistics before and after
before_stats = {
    'mean': data['fare'].mean(),
    'std': data['fare'].std(),
    'min': data['fare'].min(),
    'max': data['fare'].max(),
    'median': data['fare'].median()
}

after_stats = {
    'mean': data_cleaned['fare_capped'].mean(),
    'std': data_cleaned['fare_capped'].std(),
    'min': data_cleaned['fare_capped'].min(),
    'max': data_cleaned['fare_capped'].max(),
    'median': data_cleaned['fare_capped'].median()
}

# Create comparison table
comparison = pd.DataFrame({
    'Before_Treatment': before_stats,
    'After_Treatment': after_stats
})

print("\n📊 STATISTICAL COMPARISON:")
print(comparison.round(2))

# Calculate percentage changes
mean_change = ((after_stats['mean'] - before_stats['mean']) / before_stats['mean']) * 100
std_change = ((after_stats['std'] - before_stats['std']) / before_stats['std']) * 100
max_change = ((after_stats['max'] - before_stats['max']) / before_stats['max']) * 100

print(f"\n📈 PERCENTAGE CHANGES:")
print(f"• Mean: {mean_change:+.1f}%")
print(f"• Standard Deviation: {std_change:+.1f}%")
print(f"• Maximum Value: {max_change:+.1f}%")

# Create visualization to show the effect of outlier treatment
print("\nCreating comparison visualization...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Before treatment
axes[0].boxplot(data['fare'])
axes[0].set_title('Fare Distribution - BEFORE Outlier Treatment', fontweight='bold')
axes[0].set_ylabel('Fare ($)')
axes[0].grid(True, alpha=0.3)

# Add statistics annotation
axes[0].text(0.7, 0.95, f"Mean: ${before_stats['mean']:.2f}\nStd: ${before_stats['std']:.2f}", 
             transform=axes[0].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

# After treatment
axes[1].boxplot(data_cleaned['fare_capped'])
axes[1].set_title('Fare Distribution - AFTER Outlier Capping', fontweight='bold')
axes[1].set_ylabel('Fare ($)')
axes[1].grid(True, alpha=0.3)

# Add statistics annotation
axes[1].text(0.7, 0.95, f"Mean: ${after_stats['mean']:.2f}\nStd: ${after_stats['std']:.2f}", 
             transform=axes[1].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

plt.tight_layout()
plt.savefig('outlier_treatment_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Comparison visualization saved as 'outlier_treatment_comparison.png'")

# Step 3c: Conclude how outlier handling affected the data distribution
print("\n" + "="*50)
print("STEP 3c: CONCLUSION - EFFECT OF OUTLIER HANDLING")
print("="*50)

conclusion = f"""
Outlier handling through capping significantly improved the data distribution. 
The standard deviation decreased by {abs(std_change):.1f}%, indicating reduced variability, 
while the mean shifted {abs(mean_change):.1f}% closer to the median. 
The treatment created a more balanced distribution suitable for further analysis.
"""

print("🎯 CONCLUSION:")
print(conclusion)

print("\n" + "="*60)
print("🎉 PART B - DETECTING OUTLIERS - COMPLETED SUCCESSFULLY!")
print("="*60)

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("DATA MINING ASSIGNMENT - MISSING VALUES & OUTLIERS")
print("=" * 60)

# Load dataset
titanic = sns.load_dataset('titanic')
print("Dataset loaded: Titanic dataset")
print(f"Shape: {titanic.shape}")

# I.a. Missing values analysis
missing_count = titanic.isnull().sum()
missing_percent = (titanic.isnull().sum() / len(titanic)) * 100

missing_info = pd.DataFrame({
    'Missing_Count': missing_count,
    'Missing_Percent': missing_percent
}).sort_values('Missing_Percent', ascending=False)

print("\nI.a. MISSING VALUES ANALYSIS:")
print(missing_info[missing_info['Missing_Count'] > 0])