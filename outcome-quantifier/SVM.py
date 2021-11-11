import re
import string
import pickle
import unicodedata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams## sklearn

## sklearn
import sklearn.svm as svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

## Scipy
from scipy.sparse import csr_matrix

def clean_text(text):
    '''
        Make text lowercase, remove text in square brackets,remove links,remove punctuation
        and remove words containing numbers.
    '''
    text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text) # remove urls
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def preprocess_data(text):
    stop_words = stopwords.words('english')
    stemmer    = nltk.SnowballStemmer("english")
    text = clean_text(text)                                                     # Clean puntuation, urls, and so on
    text = ' '.join(word for word in text.split() if word not in stop_words)    # Remove stopwords
    text = ' '.join(stemmer.stem(word) for word in text.split())                # Stemm all the words in the sentence
    return text

data_folder = '../Data/reddit/title/'
positive_file_names = ['anxiety', 'depression', 'psychosis', 'stress', 'SuicideWatch']
negative_file_names = ['ask_reddit']
file_extension = '.txt'

top_grams = pd.DataFrame()
metric = pd.DataFrame(index=["Precision","Recall","Accuracy", "F1"])

for positive_file_name in positive_file_names:
    print("{0} pipeline start".format(positive_file_name))
    # Load positive dataframe
    pos_df = pd.read_csv(filepath_or_buffer=data_folder + positive_file_names[0] + file_extension, sep='❖', quotechar='⩐', header =None, names =['text'], error_bad_lines=False)
    pos_df['source'] = positive_file_name
    pos_df['label'] = 1

    neg_df = pd.read_csv(filepath_or_buffer=data_folder + negative_file_names[0] + file_extension, sep='❖', quotechar='⩐', header =None, names =['text'], error_bad_lines=False)
    ## Balance the positive and negative samples
    neg_df = neg_df.sample(n=pos_df.shape[0], random_state=1)
    neg_df['source'] = negative_file_names[0]
    neg_df['label'] = 0

    df = pd.concat([pos_df, neg_df], ignore_index=True)

    df['clean_text'] = df.text.apply(preprocess_data)

    df['split'] = np.random.choice(["train", "val", "test"], size=df.shape[0], p=[.7, .15, .15])
    x_train = df[df["split"] == "train"]
    y_train = x_train["label"]
    x_val = df[df["split"] == "val"]
    y_val = x_val["label"]

    ## Training pipeline
    tf_idf = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,3))),
        ("classifier", svm.LinearSVC(C=1.0, class_weight="balanced"))
    ])

    tf_idf.fit(x_train["clean_text"], y_train)

    ## Confidence measure
    y_pred = tf_idf.predict(x_val["clean_text"])
    print("Top n-gram claasifier F1 score: {0}".format(f1_score(y_pred, y_val)))

    coefs = tf_idf.named_steps["classifier"].coef_
    if type(coefs) == csr_matrix:
        coefs.toarray().tolist()[0]
    else:
        coefs.tolist()
    
    ## Build features for clean_text   
    feature_names = tf_idf.named_steps["tfidf"].get_feature_names()
    coefs_and_features = list(zip(coefs[0], feature_names))
    top_grams[positive_file_name] = sorted(coefs_and_features, key=lambda x: x[0], reverse=True)[:30]

    features = [x[1] for x in sorted(coefs_and_features, key=lambda x: x[0], reverse=True)[:5000]]
    for feature in features:
        df[feature] = df.clean_text.str.contains(feature).map(int)
    
    ## Build train & test set    
    X = df.drop(columns=['text', 'source', 'label', 'clean_text', 'split'])
    Y = df.label
    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, random_state=20)


    ## Train the model
    clf = svm.LinearSVC()
    
    cv_metrics = [cross_val_score(clf, X, Y, cv=5, scoring='precision').mean(),
              cross_val_score(clf, X, Y, cv=5, scoring='recall').mean(),
              cross_val_score(clf, X, Y, cv=5, scoring='accuracy').mean(),
              cross_val_score(clf, X, Y, cv=5, scoring='f1').mean()]
    metric[positive_file_names[0] + "_CV"] = cv_metrics
    
    clf.fit(X_train, Y_train)
    
    test_metrics = [precision_score(Y_test, clf.predict(X_test)),
                recall_score(Y_test, clf.predict(X_test)),
                accuracy_score(Y_test, clf.predict(X_test)),
                f1_score(Y_test, clf.predict(X_test))]
    metric[positive_file_names[0] + "_test"] = cv_metrics
    
    ## Confidence measure
    print("SVM claasifier 5-fold F1 score: {0}".format(f1_score(Y_test, clf.predict(X_test))))
    
    
    ## Save the model
    with open(positive_file_name + '.sav', 'wb') as sav:
        pickle.dump(clf, sav)
    
    print("{0} pipeline end".format(positive_file_name))
top_grams.to_csv("top_grams.csv")
metric.to_csv("metric.csv")
