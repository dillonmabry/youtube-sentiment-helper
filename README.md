# Youtube Sentiment Helper
[![Build Status](https://travis-ci.org/dillonmabry/youtube-sentiment-helper.svg?branch=master)](https://travis-ci.org/dillonmabry/youtube-sentiment-helper)
[![Python 3.4](https://img.shields.io/badge/python-3.4-blue.svg)](https://www.python.org/downloads/release/python-340/)
[![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)](https://www.python.org/downloads/release/python-350/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Determine sentiment of Youtube video per comment based analysis using Sci-kit by analyzing video comments based on positive/negative sentiment. 
Helper tool to make requests to a machine learning model in order to determine sentiment using the Youtube API.

## Install Instructions
```
pip install .
```
or PyPI (https://pypi.org/project/youtube-sentiment/)
```
pip install youtube-sentiment
```
## How to Use
Current usage:
```
import youtube_sentiment as yt
yt.video_summary(<Youtube API Key>, <Youtube Video ID>, <Max Pages of Comments>, <Sentiment Model>) 
```
or
```
python main.py <Youtube API Key> <Youtube Video ID> <Max Pages of Comments> <Sentiment Model>
```
Choices for model selection are found under the included models for setup also under project path `./models`
## Tests
```
python setup.py test
```
## To-Do
- [X] Create API to use Youtube API V3 via REST to get comments for videos
- [X] Create initial Python package
- [X] Analyze existing sentiment analysis models to select and use
- [X] Improve/enhance existing sentiment learning model
- [ ] Create deep model for sentiment
- [X] Utilize sentiment analysis to analyze Youtube video and provide analytics
- [X] Finalize Python package for project
- [ ] Fix any new bugs
- [ ] Create web based portal

## Models Available
 - lr_sentiment_basic (Basic Vectorizer/Logistic Regression model, 2 MB)
 - lr_sentiment_cv (Hypertuned TFIDF/Logistic Regression model with clean dataset, 60 MB)
 - *To-be-added* cnn_sentiment (Convolutional Neural Net model)
 - *To-be-added* cnn_sentiment (LTSM Neural Net model)

## Traditional ML Model Creation

*Why use Twitter sentiment as training?*

Twitter comments/replies/tweets are the closest existing training set to Youtube comments that are the simplest to setup. A deep autoencoder could be used to generate comments for a larger dataset (over 100k) with Youtube-esque comments but then the reliability of classifying the data would be very tricky.

**TLDR: It is the simplest and most effective to bootstrap for a traditional model**



```python
# Twitter Sentiment Analysis using traditional ML techniques
# Pipeline modeling using the following guide: https://ryan-cranfill.github.io/sentiment-pipeline-sklearn-1/
# Data processing and cleaning guide: https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90
# Stanford Twitter dataset used: http://help.sentiment140.com/for-students/

# Imports
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc, roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score
```

```python
# Dataset of 1.6m Twitter tweets
columns = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
train = pd.read_csv('stanford_twitter_train.csv', encoding='latin-1', header=None, names=columns)
test = pd.read_csv('stanford_twitter_test.csv', encoding='latin-1', header=None, names=columns)
```

```python
## Local helpers

# AUC visualization
def show_roc(title, model, test, test_labels):
    # Predict
    probs = model.predict_proba(test)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(test_labels, preds)
    roc_auc = auc(fpr, tpr)
    # Chart
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Tweet cleanser, removes hashtags, hyperlinks, cleanse data into phrases
tok = nltk.tokenize.WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
def clean_tweet(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # Tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()

# NLTK Twitter tokenizer best used for short comment-type text sets
tokenizer = nltk.casual.TweetTokenizer(preserve_case=False)

# Model reporting and scoring
def show_model_report(title, model, preds, X_test, y_test):
    print(confusion_matrix(y_test, preds))
    show_roc(title, model, X_test, y_test) # AUC report

def show_misclassified(preds, X_test, y_test):
    mis_index = np.where(y_test != preds)
    mis_df = pd.DataFrame()
    for index in mis_index:
        predict_vals = preds[index]
        features = X_test[index]
        actual_vals = y_test[index]
        mis_df = pd.DataFrame({'predicted_vals': predict_vals, 'actual_vals':actual_vals, 'features': features})
        mis_df = mis_df.dropna()
        mis_df = mis_df[mis_df.actual_vals != mis_df.predicted_vals]
    print(mis_df.head())
```

```python
# Data cleaning
cleaned_tweets = []
for tweet in train['text']:                                                                 
    cleaned_tweets.append(clean_tweet(tweet))
cleaned_df = pd.DataFrame(cleaned_tweets, columns=['text'])
cleaned_df['target'] = train.sentiment
cleaned_df.target[cleaned_df.target == 4] = 1 # rename 4 to 1 as positive label
cleaned_df = cleaned_df[cleaned_df.target != 2] # remove neutral labels
cleaned_df = cleaned_df.dropna() # drop null records
cleaned_df.to_csv('stanford_clean_twitter_train.csv',encoding='utf-8')
```

```python
# Checkpoint 1: from import clean dataset
csv = 'stanford_clean_twitter_train.csv'
df = pd.read_csv(csv,index_col=0)
```

```python
# Ensure no N/As, our dataset does have very few N/As post-cleanse, therefore omission is best course of action
df = df.dropna()
# shuffle for data subset, distributions should be even since entire dataset is split 50/50 anyway
df = df.sample(frac=1).reset_index(drop=True)
```

```python
X, y = df.text[0:500000], df.target[0:500000] # Data subset for performance reasons, though more data helps

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.10)
```

```python
# Dataset shapes post-split
print(np.shape(X_train))
print(np.shape(X_test))
print(np.unique(y_train))
```

    (450000,)
    (50000,)
    [0 1]
    

```python
# Check data distribution from train/test split should be 50/50
print(len(y_train[y_train == 0]))
print(len(y_train[y_train == 1]))
print(len(y_test[y_test == 0]))
print(len(y_test[y_test == 1]))
```

    224668
    225332
    25016
    24984
    

```python
# GridSearchCV hyperparam tuning
tfidf = TfidfVectorizer()
clf = LogisticRegression()

pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', clf)
    ])

parameters = {
    'tfidf__ngram_range': [(1,1), (1,2), (1,3)], # ngram range of tokenizer
    'tfidf__norm': ['l1', 'l2', None], # term vector normalization
    'tfidf__max_df': [0.25, 0.5, 1.0], # maximum document frequency for the CountVectorizer
    'clf__C': np.logspace(-2, 0, 3) # C value for the LogisticRegression
}

grid = GridSearchCV(pipeline, parameters, cv=3, verbose=1)
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
t0 = time.time()
grid.fit(X_train, y_train)
print("done in %0.3fs" % (time.time() - t0))
print()

print("Best score: %0.3f" % grid.best_score_)
print("Best parameters set:")
best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
```

    Performing grid search...
    pipeline: ['tfidf', 'clf']
    Fitting 3 folds for each of 81 candidates, totalling 243 fits
    

    [Parallel(n_jobs=1)]: Done 243 out of 243 | elapsed: 52.7min finished
    

    done in 3186.295s
    
    Best score: 0.803
    Best parameters set:
    	clf__C: 0.01
    	tfidf__max_df: 0.25
    	tfidf__ngram_range: (1, 3)
    	tfidf__norm: None
    

```python
# Dump model from grid search cv
joblib.dump(grid.best_estimator_, 'lr_sentiment_cv.pkl', compress=3) # compression 3 is good compromise
```


```python
# Checkpoint 2: Skip ahead here to train basic model once we know optimal hyperparameters
# Some thoughts:
# 1. Exclude analyzing tweet length since it does not enhance/degrade model performance
# 2. Tfidf performs better than standard countvector analysis which is a bit more in-depth for analysis
tfidf = TfidfVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1,3), max_df=0.25, norm=None)
clf = LogisticRegression(C=0.01)
pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', clf)
    ])
pipeline.fit(X_train, y_train)
```

    Pipeline(memory=None,
         steps=[('tfidf', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=0.25, max_features=None, min_df=1,
            ngram_range=(1, 3), norm=None, preprocessor=None, smooth_idf=True,
    ...ty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False))])


```python
scores = cross_val_score(pipeline, X_train, y_train, cv=10) # Cross-validate basic model
print(scores.mean()) # basic model tuned performs better, test with holdout set later
```

    0.8156866699389553
    


```python
# Dump model from improved
joblib.dump(pipeline, 'lr_sentiment_basic_improved.pkl', compress=3) # compression 3 is good compromise
```




    ['lr_sentiment_basic_improved.pkl']




```python
# Checkpoint 3: Post-model load comparison from existing models (lra advanced, lrb basic)
lra = joblib.load('./Models/Stanford_Twitter_Models/lr_sentiment_cv.pkl') 
lrb = joblib.load('./Models/Twitter_Simple_Models/lr_sentiment_basic.pkl')
lrb_imp = joblib.load('./Models/Stanford_Twitter_Models/lr_sentiment_basic_improved.pkl') 
```


```python
# Performance indicators for all models
lra_preds = lra.predict(X_test)
show_model_report("LR GridSearchCV", lra, lra_preds, X_test, y_test)
lrb_preds = lrb.predict(X_test)
show_model_report("LR Basic Model", lrb, lrb_preds, X_test, y_test)
lrb_imp_preds = lrb_imp.predict(X_test)
show_model_report("LR Improved Basic Model", lrb_imp, lrb_imp_preds, X_test, y_test)
```

    [[20372  4644]
     [ 4006 20978]]
    


![gridsearch](https://user-images.githubusercontent.com/10522556/49188767-f3934300-f339-11e8-9437-88bc38cfb157.png)


    [[18816  6200]
     [ 5454 19530]]
    


![basic](https://user-images.githubusercontent.com/10522556/49188766-f3934300-f339-11e8-9bf0-e04b0c68819d.png)


    [[20045  4971]
     [ 4227 20757]]
    


![improved](https://user-images.githubusercontent.com/10522556/49188765-f3934300-f339-11e8-8e41-2243a7e20f9f.png)



```python
# Display misclassified records based on predicted vs. actual with test set
show_misclassified(lra_preds, X_test, y_test)
show_misclassified(lrb_preds, X_test, y_test)
show_misclassified(lrb_imp_preds, X_test, y_test)
```

         actual_vals                                           features  \
    10           1.0  just winding down had the most perfect day tod...   
    42           1.0  oh ballarat hangs soon gonna visit mattai boi ...   
    51           0.0       walking to the library now could be swimming   
    65           0.0                                        know so sad   
    151          0.0  know right how do you screw that up but pretty...   
    
         predicted_vals  
    10                0  
    42                0  
    51                1  
    65                1  
    151               1  
         actual_vals                                           features  \
    9            1.0                                           new hair   
    10           1.0  just winding down had the most perfect day tod...   
    42           1.0  oh ballarat hangs soon gonna visit mattai boi ...   
    51           0.0       walking to the library now could be swimming   
    151          0.0  know right how do you screw that up but pretty...   
    
         predicted_vals  
    9                 0  
    10                0  
    42                0  
    51                1  
    151               1  
         actual_vals                                           features  \
    10           1.0  just winding down had the most perfect day tod...   
    51           0.0       walking to the library now could be swimming   
    285          1.0  being lazy outside sunbathing listening to all...   
    346          1.0  ve already starting retweeting too we ll get t...   
    404          1.0  wondering if you can keep up with vegas start ...   
    
         predicted_vals  
    10                0  
    51                1  
    285               0  
    346               0  
    404               0  


```python
# Holdout dataset accuracy, classification report, and display misclassified records with actual values
test_csv = 'stanford_clean_twitter_test.csv' # different twitter dataset
holdout_df = pd.read_csv(test_csv,index_col=0, encoding='latin-1')
X_h_test = holdout_df.text[0:200] # First 100 holdout set
y_h_test = holdout_df.target[0:200] # First 100 holdout set
holdout_preds = lra.predict(X_h_test)
```


```python
pipeline.score(X_h_test, y_h_test) # raw score on holdout set with improved basic model
```




    0.825




```python
# looks like we are better at classifying positive comments vs. negative
print(classification_report(y_h_test, holdout_preds))
```

                 precision    recall  f1-score   support
    
              0       0.77      0.76      0.76        74
              1       0.86      0.87      0.86       126
    
    avg / total       0.82      0.82      0.82       200
