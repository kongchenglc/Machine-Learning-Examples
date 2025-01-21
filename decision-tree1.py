# Classification Tree
# Iterative Dichotomiser 3 (ID3) algorithm

import pandas as pd
import numpy as np

# Function to calculate entropy of a dataset
def entropy(S):
    value_counts = S.value_counts(normalize=True)
    return -np.sum(value_counts * np.log2(value_counts))

# Function to calculate information gain of a feature
def information_gain(S, feature, target):
    H_S = entropy(S[target])  # Entropy of the entire dataset
    
    # Calculate weighted entropy for each value of the feature
    weighted_entropy = 0
    for value in S[feature].unique():
        subset = S[S[feature] == value]
        weighted_entropy += (len(subset) / len(S)) * entropy(subset[target])
    
    # Information gain is the reduction in entropy after the split
    return H_S - weighted_entropy

# Function to build the decision tree recursively
def build_tree(S, features, target):
    # If all instances have the same label, return the label
    if S[target].nunique() == 1:
        return S[target].iloc[0]
    
    # If there are no features left, return the most frequent label
    if not features:
        return S[target].mode()[0]
    
    # Find the feature with the highest information gain
    best_feature = max(features, key=lambda f: information_gain(S, f, target))
    
    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]
    
    # For each value of the best feature, split the data and recurse
    for value in S[best_feature].unique():
        subset = S[S[best_feature] == value]
        tree[best_feature][value] = build_tree(subset, remaining_features, target)
    
    return tree

# Example dataset: Play Tennis
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Overcast'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot'],
    'Humidity': ['High', 'High', 'High', 'High', 'Low', 'Low', 'Low', 'High', 'Low', 'Low'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Weak', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes']
}

# Create DataFrame from the data
df = pd.DataFrame(data)

# Define the target variable and features
target = 'PlayTennis'
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

print("Source Data: \n", df)

# Build the classification tree
tree = build_tree(df, features, target)

# Print the resulting decision tree
print("\nDecision Tree:", tree)
