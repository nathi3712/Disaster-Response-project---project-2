import sys

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import trigrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import joblib

from sklearn import metrics
from sqlalchemy import create_engine
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import  f1_score,precision_score,recall_score,accuracy_score,make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import TruncatedSVD
import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessagesTables',engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize the text - separating the words
    words = word_tokenize(text)
    #removing stop words such as (is,the, if...)
    words = [w for w in words if w not in                                                       stopwords.words("english")]
    # Reduce words to their stems
    words = [PorterStemmer().stem(w) for w in words]
    # Reducing words to their root form
    words = [WordNetLemmatizer().lemmatize(w) for w in words]

    return words



#credit to siddhardhan from youtube on this helper code and some research on stackoverflow
# Below I am making a function for the f1_score, precision and recall to run through the data of each column

def Precision_recall_f1_score(true_value, pred_value):
    '''Making a function to put the Classification report in table format.
       Input actual values and predicted values
       Output is the classification report (shown as weighted, macro and micro)
    '''
    precision_weighted = precision_score(true_value, pred_value, average = "weighted" )
    recall_weighted = recall_score(true_value, pred_value, average = "weighted")
    f1_weighted = f1_score(true_value, pred_value, average = "weighted")
    precision_macro = precision_score(true_value, pred_value, average = "macro" )
    recall_macro = recall_score(true_value, pred_value, average = "macro")
    f1_macro = f1_score(true_value, pred_value, average = "macro")
    precision_micro = precision_score(true_value, pred_value, average = "micro" )
    recall_micro = recall_score(true_value, pred_value, average = "micro")
    f1_micro = f1_score(true_value, pred_value, average = "micro")
    return {"Precision_weighted": precision_weighted, "Recall_weighted":                                 recall_weighted,"F1_Score_weighted": f1_weighted, 
            "Precision_macro": precision_macro, "Recall_macro":                                         recall_macro,"F1_Score_macro": f1_macro, 
            "Precision_micro": precision_micro, "Recall_micro":                                         recall_micro,"F1_Score_micro": f1_micro}


def build_model():
    pipeline = Pipeline([('vect', TfidfVectorizer(tokenizer = tokenize, use_idf = True,                    smooth_idf = True, sublinear_tf = False)),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__max_df': (0.5, 1.0),
        'clf__estimator__n_estimators': [5, 10, 20],
        'clf__estimator__min_samples_split': [2]
    }

    model = GridSearchCV(pipeline, param_grid = parameters, cv = 5, refit = True,                        return_train_score = True,verbose=5)
    
    return model


def evaluate_model(model, X_test, y_test, category_names):
    y_predict_test2 = model.predict(X_test)
    Test_Report2 = []
    for m, column in enumerate(y_test.columns):
        Report = Precision_recall_f1_score(y_test.loc[:,column], y_predict_test2[:, m])
        Test_Report2.append(Report)
    Test_Report2_df = pd.DataFrame(Test_Report2)
    print(Test_Report2_df.mean())
    print(Test_Report2_df)
    

def save_model(model, model_filepath):
    model = model.best_estimator_
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()