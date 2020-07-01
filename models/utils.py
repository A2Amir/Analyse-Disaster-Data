
#!/usr/bin/env python
# coding: utf-8



# - Import Python libraries


import nltk
nltk.download(['punkt','wordnet','averaged_perceptron_tagger','stopwords'])
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import re



from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import fbeta_score,classification_report,make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from joblib import dump

import pandas as pd
import sqlite3
import numpy as np
import os 


def split_data(X, y):
    '''
    to split data into the train and test set 
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test
    

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
     A custom transformer that extracts starting verb
    """
    
    def starting_verb(self, text):
        
        # tokenize by sentence
        sentence_list = sent_tokenize(text)
        
        for sentence in sentence_list:
            
            # tokenize each sentence into words and tag part of speech
            pos_tags = pos_tag(tokenize(sentence))
            
            if len(pos_tags) != 0:
                # index pos_tags to get the first word and part of speech tag
                first_word, first_tag = pos_tags[0]

                # return true if the part of speech is an apporpriate verb
                if first_tag in ['VB','VBP']:
                    return True
            
        return False
        
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        To extract whether text starts with a verb
        """
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
        


class CaseNormalizer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that converts all text to lowercase
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        to convert all text to lowercase
        """
  
        return pd.Series(X).apply(lambda x: x.lower()).values
    
def tokenize(text):
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ',text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()


    # Remove stop words
    # tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='n').strip()
        #I passed in the output from the previous noun lemmatization step. This way of chaining procedures is very common.
        clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
        #It is common to apply both, lemmatization first, and then stemming.
        #clean_tok =PorterStemmer().stem(clean_tok)
        clean_tokens.append(clean_tok)
    
    return clean_tokens





def load_data(f1_dir):
    
    # Load dataset from database 
    db = sqlite3.connect(f1_dir)
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()[0][0]
    df = pd.read_sql_query('SELECT * FROM '+tables,db)

    X = df['message']
    y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return df, X, y, category_names




# The machine learning pipeline
# This machine pipeline takes in the `message` column as input and output classification results on the other 36 categories in the dataset. 
def get_pipeline(clf=RandomForestClassifier()):
    
 
    pipeline = Pipeline([
                        ('features',FeatureUnion([
                                                 ('text-pipeline',Pipeline([
                                                                            ('lowercase', CaseNormalizer()),
                                                                            ('vect', CountVectorizer(tokenizer= tokenize)),
                                                                            ('tfidf', TfidfTransformer())
                                                                           ])),
                                                 ('starting-verb',StartingVerbExtractor())
                                                 ])),
                        ('clf', MultiOutputClassifier(clf))
                       ])
    return pipeline



# To report the fbeta score for the whole model, precision and recall for each output category of the dataset by iterating through the columns and calling sklearn's `classification_report` on each.
def get_fbeta_score(y_true, y_pred):

    """
    Compute F_beta score, the weighted harmonic mean of precision and recall

    Parameters
    ----------
    y : Pandas Dataframe
        y true
    y_pred : array 
        y predicted 

    Returns
    -------
    fbeta_score : float
    """
    score_list = []
    
        
    for index, col in enumerate(y_true.columns):
        error = fbeta_score(y_true[col], y_pred[:,index],1,average='weighted')
        score_list.append(error)
        
    fb_score_numpy = np.asarray(score_list)
    fb_score_numpy = fb_score_numpy[fb_score_numpy<1]
    fb_score = np.mean(fb_score_numpy)
    
    return fb_score




# Use grid search to find better parameters. 
def build_model(X_train, y_train, clf = RandomForestClassifier()):
    
    pipeline = get_pipeline(clf)

    # specify parameters for grid search
    parameters = {  #'clf__estimator__min_samples_split': [2, 4],
                    #'clf__estimator__criterion': ['log2', 'auto', 'sqrt', None],
                    # 'features__text-pipeline__tfidf__use_idf' : [True, False],
                    'clf__estimator__criterion': ['gini', 'entropy'],
                    #'clf__estimator__max_depth': [None, 25, 50],
                  }


    make_score= make_scorer(get_fbeta_score,greater_is_better=True)
        
     # create grid search object
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring=make_score)
    cv.fit(X_train,y_train)
    
    return cv

def dump_model(model):   
    
    # Export your model as a pickle file
    model_path = os.getcwd()+'\\model.pkl'
    dump(model, open(model_path, 'wb') ,compress = 1)
