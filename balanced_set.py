import random 

import pandas as pd 
import numpy as np 

# Load in the full dataframe
df = pd.read_csv('clean_dataset.csv')

# Separate true positives and true negatives
bully_df = df[df['Insult'] == 1]
normal_df = df[df['Insult'] == 0]

# Reset index after separating
bully_df.reset_index(inplace=True, drop=True)
normal_df.reset_index(inplace=True, drop=True)

# Get N + 10 number of true negatives
# Sample without replacement
N = len(bully_df) + 10 
random_numbers = random.sample(range(0, len(normal_df)), N)
new_normal_df = normal_df.iloc[random_numbers]

# Reset index
new_normal_df.reset_index(inplace=True, drop=True)

# Combine the dataframes
balanced_df = pd.concat([bully_df, new_normal_df])

print(balanced_df.head(30))