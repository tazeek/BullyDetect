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

print(normal_df.head())