import time
import os

import numpy as np
import pandas pd

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, log_loss, brier_score_loss

os.system('cls')

# Load the dataset here
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']