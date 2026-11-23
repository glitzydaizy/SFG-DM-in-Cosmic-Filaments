import pandas as pd

# Replace with the actual filename of your final table
df = pd.read_csv('results/d1213_sfg_sfr_v2.csv')

print(f"Total matched galaxies (rows): {len(df)}")
print("\nColumn headers:")
print(list(df.columns))

print("\nFirst 3 rows preview:")
print(df.head(3))

print("\nSummary statistics (min, median, mean, max):")
print(df.describe())

