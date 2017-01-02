# BullyDetect (Techniques to Detect Cyberbully)

My final year project at Multimedia University, Cyberjaya. It involves Natural Language Processing to detect cyberbullies using a combination of supervised and unsupervised learning based on the text comment. 

## Supervised Learning

The dataset used for classification is from **Kaggle** and the following supervised machine learning algorithms
were used:

- Random Forest (100 Trees)
- Naive Bayes (Gaussian Model)
- Support Vector Machines (Linear SVC)

[KAGGLE DATASET LINK](https://www.kaggle.com/c/detecting-insults-in-social-commentary/data)

**EXTRA**: Currently trying out the following approaches since the main phases are over:
- Fine-tuning parameters of machine learning approaches
- Using XGBoost for a try out

## Unsupervised Learning

The framework used is **Word2Vec Skip-Gram** model. The model was trained using comments from the **Reddit** corpus, from **January 2015**
to **May 2015**.  Also, K-Means Clustering was used in conjunction with Word2Vec. The skip-gram model is shown below:

![Skip-gram model][sg-w2v]

[REDDIT CORPUS LIST](https://archive.org/download/2015_reddit_comments_corpus/reddit_data/)

## Methods used

Some of the main methods used are:

- **Average Words**: The most basic approach. Add the feature vectors of words, then divide by the total number of words.
- **Mean Similarity**: Finding the feature vectors of words that are above a mean cosine similarity. This is done by finding the *top-n* words, and averaging their mean similarity. This is done word-by-word.
- **Word Feature**: Using the mean feature of each specific word, provided it is in the model.
- **Clustering Word Vectors**: Using K-Means Clustering to cluster a group of words together.

Some of the above methods can be combined using the *TF-IDF* from the Kaggle Dataset

## Evaluation and Results

The following evaluation metrics were used after being cross-validated with **Stratified 10 Fold Sampling**: *Accuracy*, *Precision*, *False Positive Rate (FPR)*, *Area Under ROC*, *Log Loss*, *Brier Score Loss* and *Run-Time Prediction*. Due to the dataset being negatively skewed (about 75% non-bully comments), a lot of importance were put on **Precision**, **FPR**, **Brier Score Loss**, and **Run-Time Prediction**. The results are divided into two jupyter notebooks, based on two different datasets:

- [Balanced Dataset][bala]: Using an even number of bully and non-bully comments
- [Imbalanced Dataset][imba]: Using the full dataset

Also, for evaluation of Word2Vec can be found [here][w2v-eval]

## Tools Used

**Python 3.5+** was used as the scripting language, while **MongoDB** was used to store the comments from reddit. Some of the main libraries used:

- [Gensim][gensim]: For Word2Vec.
- [Scikit-learn][sklearn]: For Machine Learning and Evaluation Metrics.
- [Regex][regex]: For handling character-level expressions in text.


[sg-w2v]: http://sebastianruder.com/content/images/2016/02/skip-gram.png
[gensim]: https://radimrehurek.com/gensim/models/word2vec.html
[sklearn]: http://scikit-learn.org/stable/index.html
[regex]: https://docs.python.org/3/library/re.html
[w2v-eval]: https://github.com/tazeek/BullyDetect/blob/master/Python%20Notebooks/Word2Vec%20Evaluation.ipynb
[bala]: https://github.com/tazeek/BullyDetect/blob/master/Python%20Notebooks/Results%20Visualisation%20(Balanced%20Dataset).ipynb
[imba]: https://github.com/tazeek/BullyDetect/blob/master/Python%20Notebooks/Results%20Visualisation%20(Imbalanced%20Dataset).ipynb