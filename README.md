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
to **May 2015**.  Also, K-Means Clustering was used in conjunction with Word2Vec

[REDDIT CORPUS LIST](https://archive.org/download/2015_reddit_comments_corpus/reddit_data/)


# Libraries