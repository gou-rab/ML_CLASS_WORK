import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('gini1.csv')
print("Dataset loaded:")
print(df.head())
print("\nClass distribution:")
print(df['Decision'].value_counts())
print()

# Auto-detect column names (handles spaces)
income_col = df.columns[0]  # First column: Income
lawn_col = df.columns[1]    # Second column: Lawn Size/Size
target_col = df.columns[2]  # Decision
features = [income_col, lawn_col]
print(f"Using columns: {features[0]}, {features[1]}, {target_col}")
print()

def gini_impurity(y):
    """Calculate Gini impurity for a label series."""
    if len(y) == 0:
        return 0.0
    p = y.value_counts(normalize=True)
    return 1 - np.sum(p**2)

def best_split_gini(df, feature_col, target_col):
    """Find best split threshold for a numerical feature using Gini."""
    df_sorted = df[[feature_col, target_col]].sort_values(feature_col).copy()
    best_gini = float('inf')
    best_threshold = None
    best_left_count = best_right_count = 0
    
    parent_gini = gini_impurity(df[target_col])
    
    for i in range(1, len(df_sorted)):
        threshold = (df_sorted.iloc[i-1][feature_col] + df_sorted.iloc[i][feature_col]) / 2
        
        left_mask = df_sorted[feature_col] < threshold
        right_mask = df_sorted[feature_col] >= threshold
        
        left_labels = df_sorted.loc[left_mask, target_col]
        right_labels = df_sorted.loc[right_mask, target_col]
        
        n_left, n_right = len(left_labels), len(right_labels)
        if n_left == 0 or n_right == 0:
            continue
            
        gini_left = gini_impurity(left_labels)
        gini_right = gini_impurity(right_labels)
        weighted_gini = (n_left / len(df_sorted)) * gini_left + (n_right / len(df_sorted)) * gini_right
        
        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_threshold = threshold
            best_left_count = n_left
            best_right_count = n_right
    
    gini_gain = parent_gini - best_gini
    return best_threshold, best_gini, gini_gain, best_left_count, best_right_count

# Calculate parent Gini
parent_gini = gini_impurity(df[target_col])
print(f"Parent Gini Impurity: {parent_gini:.4f}")
print()

# Find best splits for both features
print("Best splits for each feature:")
for feature in features:
    threshold, post_gini, gain, left_n, right_n = best_split_gini(df, feature, target_col)
    print(f"{feature}:")
    print(f"  Best threshold: {threshold:.2f}")
    print(f"  Post-split Gini: {post_gini:.4f}")
    print(f"  Gini Gain: {gain:.4f}")
    print(f"  Left: {left_n} samples, Right: {right_n} samples")
    print()

# Determine overall best feature to split on
best_feature_data = max([(f, best_split_gini(df, f, target_col)[2]) for f in features], key=lambda x: x[1])
best_feature = best_feature_data[0]
print(f"Overall best feature to split: {best_feature}")

# Show split details for best feature
best_thresh, _, _, _, _ = best_split_gini(df, best_feature, target_col)
left_mask = df[best_feature] < best_thresh
print(f"\nSplit on {best_feature} < {best_thresh:.2f}:")
print("Left node:", df.loc[left_mask, target_col].value_counts().to_dict())
print("Right node:", df.loc[~left_mask, target_col].value_counts().to_dict())
