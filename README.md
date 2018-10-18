# Youtube Sentiment Helper
Determine sentiment of Youtube video per comment based analysis using Sci-kit by analyzing video comments based on positive/negative sentiment. 
Helper tool to make requests to a machine learning model in order to determine sentiment using the Youtube API.

## Install Instructions
`pip install .`
## How to Use
Current usage:
```
python main.py <Youtube API Key> <Youtube video ID> <Max Pages of Comments>
```
or
```
import youtube_sentiment as yt
yt.process_video_comments(<Youtube API Key>, <Youtube video ID>, <Max Pages of Comments>) 
```
## Tests
```
python setup.py test
```
## To-Do
- [X] Create API to use Youtube API V3 via REST to get comments for videos
- [X] Analyze existing sentiment analysis models to select and use
- [ ] Improve/enhance sentiment learning model
- [ ] Utilize sentiment analysis to analyze Youtube video and provide analytics
- [ ] Finalize and create Python package for project 
- [ ] Create web based portal

## Traditional ML Model Creation

```python
# Develop sentiment analysis classifier using traditional ML models
# Feature union and pipeline modifications using the following guide: 
# https://ryan-cranfill.github.io/sentiment-pipeline-sklearn-1/

# Imports
import numpy as np
import time
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
```


```python
# Dataset of ~100,000 Twitter tweets with corresponding labels (positive 1, negative 0)
# Dataframe columns: ID, Sentiment (score), SentimentText
train = pd.read_csv('twitter_train.csv', encoding='latin-1')
test = pd.read_csv('twitter_test.csv', encoding='latin-1')
```


```python
X, y = train.SentimentText, train.Sentiment

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```


```python
# Dataset shapes post-split
print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))
print(np.unique(y_train))
print(np.unique(y_test))
```

    (74991,)
    (24998,)
    (74991,)
    (24998,)
    [0 1]
    [0 1]
    


```python
# NLTK Twitter tokenizer best used for short comment-type text sets
tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
```


```python
# Replace @ mentions with generic token
import re

def replace_mentions(text):
    return re.sub(r'@[\w_-]+', 'MENTION', text)
```


```python
from sklearn.preprocessing import FunctionTransformer

def pipeline_function(function, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            return [function(i) for i in list_or_series]
        else: # if it's not active, just pass it right back
            return list_or_series
    return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active':active})
```


```python
def get_post_length(text):
    return len(text)
```


```python
# Add custom features (namely length, since longer comments/posts will most likely be either positive or negative)
# Utilize FeatureUnion class

def reshape_a_feature_column(series):
    return np.reshape(np.asarray(series), (len(series), 1))

def pipeline_feature(function, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            processed = [function(i) for i in list_or_series]
            processed = reshape_a_feature_column(processed)
            return processed
        # If a feature is deactivated, we're going to just return a column of zeroes.
        else:
            return reshape_a_feature_column(np.zeros(len(list_or_series)))
```


```python
# Hyperparameter tuning (Simple model)
cvect = CountVectorizer(tokenizer=tokenizer.tokenize) 
clf = LogisticRegression()

pipeline = Pipeline([
        ('cvect', cvect),
        ('clf', clf)
    ])

parameters = {
    'cvect__ngram_range': [(1,1), (1,2), (1,3)], # ngram range of tokenizer
    'cvect__max_df': [0.25, 0.5, 1.0], # maximum document frequency for the CountVectorizer
    'clf__C': np.logspace(-1, 0, 2) # C value for the LogisticRegression
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
    pipeline: ['cvect', 'clf']
    Fitting 3 folds for each of 18 candidates, totalling 54 fits
    

    [Parallel(n_jobs=1)]: Done  54 out of  54 | elapsed: 13.3min finished
    

    done in 826.190s
    
    Best score: 0.783
    Best parameters set:
    	clf__C: 1.0
    	cvect__max_df: 0.5
    	cvect__ngram_range: (1, 3)
    


```python
# Dump model from grid search cv
joblib.dump(grid.best_estimator_, 'lr_sentiment_cv.pkl', compress=1)
```




    ['lr_sentiment_cv.pkl']




```python
# Hyperparameter tuning (Slightly advanced model)
tokenizer_lowercase = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=False)
tokenizer_lowercase_reduce_len = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
tokenizer_uppercase = nltk.casual.TweetTokenizer(preserve_case=True, reduce_len=False)
tokenizer_uppercase_reduce_len = nltk.casual.TweetTokenizer(preserve_case=True, reduce_len=True)

parameters = {
    'mentions_replace__kw_args': [{'active':False}, {'active':True}], # genericizing mentions on/off
    'features__post_length__kw_args': [{'active':False}, {'active':True}], # adding post length feature on/off
    'features__cvect__ngram_range': [(1,1), (1,2), (1,3)], # ngram range of tokenizer
    'features__cvect__tokenizer': [tokenizer_lowercase.tokenize, # differing parameters for the TweetTokenizer
                                        tokenizer_lowercase_reduce_len.tokenize,
                                        tokenizer_uppercase.tokenize,
                                        tokenizer_uppercase_reduce_len.tokenize,
                                        None], # None will use the default tokenizer
    'features__cvect__max_df': [0.25, 0.5], # maximum document frequency for the CountVectorizer
    'clf__C': np.logspace(-2, 0, 3) # C value for the LogisticRegression
}

grid = GridSearchCV(pipeline, parameters, cv=5, verbose=1)
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


```python
# Dump model from grid search cv
joblib.dump(grid.best_estimator_, 'lr_sentiment_adv_cv.pkl', compress=1)
```


```python
grid.predict(["that was the best movie ever"])
```




    array([1], dtype=int64)


