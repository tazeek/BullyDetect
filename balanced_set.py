import pandas as pd 

# Load in the full dataframe
df = pd.read_csv('clean_dataset.csv')

# Separate true positives and true negatives
bully_df = df[df['Insult'] == 1]
normal_df = df[df['Insult'] == 0]

# Reset index after separating
bully_df.reset_index(inplace=True, drop=True)
normal_df.reset_index(inplace=True, drop=True)

# Get N number of true negatives
# Sample without replacement
N = len(bully_df)

new_normal_df = normal_df.sample(n=N, replace=False)

# Drop index
new_normal_df.reset_index(inplace=True, drop=True)

print(new_normal_df.head(10))