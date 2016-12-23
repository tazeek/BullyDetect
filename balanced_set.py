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

comments_array = np.random.choice(normal_df['Comment'], N, replace=False)
new_normal_df = normal_df.sample(n=N, replace=False)

# Reset index and drop duplicates, if any
print(len(new_normal_df))
new_normal_df.drop_duplicates(inplace=True)
new_normal_df.reset_index(inplace=True, drop=True)

print(len(new_normal_df))