# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:47:25 2020

@author: wegma
"""

from matplotlib import pyplot
import pandas as pd
import numpy as np
import re
import string


from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer


from gs_connection import (gs_connection)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier




from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import spacy

import gensim.parsing.preprocessing as gsp
from gensim import utils

def lemmatizer(text):
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)

def clean_text(s):
    filters = [gsp.strip_tags,
               gsp.strip_punctuation,
               gsp.strip_multiple_whitespaces,
               gsp.strip_numeric]
    s = re.sub(r'http\S+', '', s)
    s = s.lower()
    s = utils.to_unicode(s)
    s = utils.deaccent(s)
    for f in filters:
        s = f(s)
    return s

def process_clean_stopwords(text, nltk_stopwords):
    wordList = word_tokenize(text)
    remove_stopwords = [
        word for word in wordList if word not in nltk_stopwords]
    untokens = TreebankWordDetokenizer().detokenize(remove_stopwords)
    return untokens


def create_model(model_selected, all_posts, eval_error = "merror" , eval_loss = "mlogloss"):
    tfidfconverter = TfidfVectorizer(max_features= 1500, min_df=1, max_df=0.9)
    X = tfidfconverter.fit_transform(all_posts["lemma_text"]).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        all_posts["Genre"],
                                                        test_size=0.20,
                                                        random_state=142)

    classifier = model_selected
    #classifier.fit(X_train, y_train, verbose = True)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    classifier.fit(X_train, y_train, eval_metric=[eval_error, eval_loss], eval_set=eval_set, verbose=True)
    # make predictions for test data
    y_predict = classifier.predict(X_test)

    # retrieve performance metrics
    results = classifier.evals_result()
    epochs = len(results['validation_0'][eval_error])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0'][eval_loss], label='Train')
    ax.plot(x_axis, results['validation_1'][eval_loss], label='Test')
    ax.legend()
    pyplot.ylabel('Log Loss')
    pyplot.title('Model Log Loss')
    pyplot.show()
    # plot classification error
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0'][eval_error], label='Train')
    ax.plot(x_axis, results['validation_1'][eval_error], label='Test')
    ax.legend()
    pyplot.ylabel('Classification Error')
    pyplot.title('Model Classification Error')
    pyplot.show()

    print()
    print(confusion_matrix(y_test, y_predict))
    print()
    print(classification_report(y_test, y_predict))

    return classifier, tfidfconverter

stopwords_nltk = set(stopwords.words('english'))
#exclude = set(string.punctuation)
#lemma = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

path_movies = "/Users/wegma/OneDrive/Documents/personal/movies_classify/"
movies = pd.read_csv(path_movies + "wiki_movie_plots_deduped.csv")

movies.shape

movies.columns

movies.head()

movies.Genre.value_counts()
movies = movies[movies["Genre"] != "unknown"]

movies = movies[["Title", "Plot", "Genre"]]
movies = movies.drop_duplicates()

#There are 1399 repeated movie names... let keep only one of teach
sum(movies["Title"].value_counts()>1)

#some movies might have same name, lets drop them
movies = movies.drop_duplicates(subset="Title")



movies["Genre"] = movies["Genre"].replace("", np.nan)
movies["Genre"] = movies["Genre"].replace(" ", np.nan)
movies = movies.dropna(subset =["Genre"])
movies = movies.reset_index(drop=True)

movies2 = pd.concat([pd.Series(row['Title'], row['Genre'].split(',')) for _, row in movies.iterrows()]).reset_index()

movies2.columns = ["Genre", "Title"]
movies2["Genre"] = movies2["Genre"].replace(" ", "", regex = True)


movies3 = pd.merge(movies[["Title", "Plot"]], movies2, on ="Title", how = "inner")
movies3 = movies3.drop_duplicates()


#lets keep only the top 5 genres
movies3.Genre.value_counts()[0:5]
movies3 = movies3[movies3["Genre"].isin(movies3.Genre.value_counts()[0:5].index)]



#we need to lemmatize all the plots
movies3["content_clean"] = movies3["Plot"].apply(lambda x: clean_text(x))
movies3["content_clean"] = movies3["content_clean"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_nltk)]))
movies3 = movies3.dropna(subset=["content_clean"])


movies3["content_clean"] = movies3["content_clean"].replace("", np.nan)
movies3["content_clean"] = movies3["content_clean"].replace(" ", np.nan)
movies3 = movies3.dropna(subset=["content_clean"])

movies3["lemma_text"] = movies3["content_clean"].apply(lambda x: lemmatizer(x))




create_model(model_selected = XGBClassifier(n_estimators= 200,
                                            max_depth = 6,
                                            gamma = 0.01,
                                            reg_lambda = 5,
                                            n_jobs = -1) ,
             all_posts= movies3)

#what about undersampling?

len_undersampling = movies3["Genre"].value_counts()[-1]

list_genres=list(set(movies3["Genre"]))

movies_under = []
for genre in list_genres:
     sub = movies3[movies3["Genre"]== genre]
     sub = sub.sample(n=len_undersampling)
     movies_under.append(sub)

movies_under = pd.concat(movies_under)
movies_under["Genre"].value_counts()


create_model(model_selected = XGBClassifier(n_estimators= 200,
                                            max_depth = 6,
                                            gamma = 0.01,
                                            reg_lambda = 5,
                                            n_jobs = -1) ,
             all_posts= movies_under)



#onevrest

def create_onevrest_df(movies_df, genre, multiplier = 3):
     genre_df = movies3[movies3["Genre"]== genre]
     other_df = movies3[movies3["Genre"]!= genre]
     other_df = other_df.sample(n=int(len(genre_df)*multiplier))
     other_df["Genre"] = "other"
     return pd.concat([genre_df,other_df])

set(movies3["Genre"])

action_df = create_onevrest_df(movies3, "action")
drama_df = create_onevrest_df(movies3, "drama", multiplier=1)
horror_df = create_onevrest_df(movies3, "horror")
romance_df = create_onevrest_df(movies3, "romance")
comedy_df = create_onevrest_df(movies3, "comedy", multiplier=2)

action_df["Genre"].value_counts()




create_model(model_selected = XGBClassifier(n_estimators= 200,
                                            max_depth = 4,
                                            gamma = .1,
                                            reg_lambda = 1,
                                            scale_pos_weight = .05,
                                            n_jobs = -1) ,
             all_posts= action_df,
             eval_error = "error" , eval_loss = "logloss")
all_posts=action_df
tfidfconverter = TfidfVectorizer(max_features= 1500, min_df=1, max_df=0.9)
X = tfidfconverter.fit_transform(all_posts["lemma_text"]).toarray()
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    all_posts["Genre"],
                                                    test_size=0.20,
                                                    random_state=142)

classifier = XGBClassifier(n_estimators= 200,
                                            max_depth = 6,
                                            gamma = 0.01,
                                            reg_lambda = 5,
                                            n_jobs = -1)
#classifier.fit(X_train, y_train, verbose = True)
eval_set = [(X_train, y_train), (X_test, y_test)]
classifier.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# make predictions for test data
y_predict = classifier.predict(X_test)



parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6, 8, 10, 12, 15],
              'subsample': [0.6, 0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [10, 50, 100], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [142],
              'n_jobs':[-1]}


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV


clf = GridSearchCV(XGBClassifier, parameters, n_jobs=5,
                   cv=StratifiedKFold(movies3["Genre"], shuffle=True),
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(movies3["lemma_text"], movies3["Genre_cat"])




from scipy import stats
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,roc_auc_score
from sklearn.model_selection import cross_validate


clf_xgb = XGBClassifier(objective = 'binary:logistic')

param_dist = {'n_estimators': stats.randint(10, 200),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 8, 10, 12, 15],
              'colsample_bytree': stats.uniform(0.5, 0.9),
             }

numFolds = 5
kfold_5 = StratifiedKFold(shuffle = True)

clf = RandomizedSearchCV(clf_xgb, 
                         param_distributions = param_dist,
                         cv = kfold_5,  
                         n_iter = 5, # you want 5 here not 25 if I understand you correctly 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 3, 
                         n_jobs = -1)

tfidfconverter = TfidfVectorizer(max_features= 1500, min_df=1, max_df=0.9)
X = tfidfconverter.fit_transform(movies3["lemma_text"]).toarray()
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    movies3["Genre"],
                                                    test_size=0.20,
                                                    random_state=142)



clf.fit(X_train, y_train)




movies3["Genre_cat"]= pd.factorize(movies3["Genre"])[0] + 1



tfidfconverter = TfidfVectorizer(max_features= 1500, min_df=1, max_df=0.9)
X = tfidfconverter.fit_transform(movies3["lemma_text"]).toarray()
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    movies3["Genre"],
                                                    test_size=0.20,
                                                    random_state=142)

y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)

classifier = XGBClassifier(n_estimators= 100,
                                            max_depth = 20,
                                            n_jobs = -1, 
                                            verbosity= 1)
#classifier.fit(X_train, y_train, verbose = True)
classifier.fit(X_train, y_train, verbose=True)
# make predictions for test data
y_predict = classifier.predict(X_test)


print(confusion_matrix(y_test, y_predict))
print()
print(classification_report(y_test, y_predict))