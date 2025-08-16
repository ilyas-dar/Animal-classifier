# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set matplotlib to display plots inline (optional, for Jupyter or similar)


# Load the CSV file
csv_path = 'L:\\ilyas\\myFirstModel\\animal_data.csv'
try:
    df = pd.read_csv(csv_path)
    print("Successfully loaded animal_data.csv\n")
except FileNotFoundError:
    print(f"Error: {csv_path} not found. Please run train_animal_classifier.py first.")
    exit()

# Step 1: Print the data
print("Full Dataset (first 5 rows):")
print(df.head(), "\n")

print("Dataset Shape (rows, columns):", df.shape, "\n")

print("Images per Animal:")
print(df['animal'].value_counts(), "\n")

print("Summary Statistics for Features:")
print(df.describe(), "\n")

print("Pig vs. Rabbit Feature Means (first 5 histogram features):")
print(df[df['animal'] == 'pig'][['hist_0', 'hist_1', 'hist_2', 'hist_3', 'hist_4']].mean())
print(df[df['animal'] == 'rabbit'][['hist_0', 'hist_1', 'hist_2', 'hist_3', 'hist_4']].mean(), "\n")

# Step 2: Create Graphs
# Set seaborn style for better visuals
sns.set(style="whitegrid")

# Graph 1: Scatter Plot of hist_0 vs. hist_1
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='hist_0', y='hist_1', hue='animal', style='animal', s=100)
plt.title('Scatter Plot of hist_0 vs. hist_1 by Animal')
plt.xlabel('Histogram Feature (hist_0)')
plt.ylabel('Histogram Feature (hist_1)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('L:\\ilyas\\myFirstModel\\scatter_hist0_hist1.png')
plt.close()
print("Saved scatter plot: L:\\ilyas\\myFirstModel\\scatter_hist0_hist1.png")

# Graph 2: Histogram of hist_0 for Pig vs. Rabbit
plt.figure(figsize=(10, 6))
for animal in ['pig', 'rabbit']:
    sns.histplot(data=df[df['animal'] == animal], x='hist_0', label=animal, alpha=0.5, bins=20)
plt.title('Histogram of hist_0 for Pig vs. Rabbit')
plt.xlabel('Histogram Feature (hist_0)')
plt.ylabel('Count')
plt.legend()
plt.savefig('L:\\ilyas\\myFirstModel\\hist_pig_rabbit.png')
plt.close()
print("Saved histogram: L:\\ilyas\\myFirstModel\\hist_pig_rabbit.png")

# Graph 3: Box Plot of hist_0 across all animals
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='animal', y='hist_0')
plt.title('Box Plot of hist_0 by Animal')
plt.xlabel('Animal')
plt.ylabel('Histogram Feature (hist_0)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('L:\\ilyas\\myFirstModel\\boxplot_hist0.png')
plt.close()
print("Saved box plot: L:\\ilyas\\myFirstModel\\boxplot_hist0.png")

print("\nGraphs saved. Check L:\\ilyas\\myFirstModel\\ for images.")
print("To improve accuracy, ensure animal features (e.g., hist_0, hist_1) differ significantly.")