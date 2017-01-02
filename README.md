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
to **May 2015**.  Also, K-Means Clustering was used in conjunction with Word2Vec.

[REDDIT CORPUS LIST](https://archive.org/download/2015_reddit_comments_corpus/reddit_data/)

## Tools Used

**Python 3.5+** was used as the scripting language, while **MongoDB** was used to store the comments from reddit. Some of the main libraries used:

- [Gensim][gensim]: For Word2Vec.
- [Scikit-learn][sklearn]: For Machine Learning and Evaluation Metrics.
- [Regex][regex]: For handling character-level expressions in text.


[gensim]: https://radimrehurek.com/gensim/models/word2vec.html
[sklearn]: http://scikit-learn.org/stable/index.html
[regex]: https://docs.python.org/3/library/re.html