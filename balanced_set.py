import pandas as pd 
import numpy as np 

# Load in the full dataframe
df = pd.read_csv('clean_dataset.csv')

# Separate true positives and true negatives
bully_df = df[df['Insult'] == 1]
normal_df = df[df['Insult'] == 0]

print(len(bully_df))
print(len(normal_df))