import json
import plotly
import pandas as pd

import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessagesTables', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    #looking as how much the weather played a role
    Weather_1 = df[df['weather_related']==1].groupby('genre').count()['message']
    Weather_0 = df[df['weather_related']==0].groupby('genre').count()['message']
    genre_names = list(Weather_1.index)

    #Catergories involved of the disaster messages 
    category_columns = df.drop(['id', 'message', 'original', 'genre'], axis = 1).columns
    category_values = (df.drop(['id', 'message', 'original', 'genre'], axis = 1) != 0).sum().values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
            
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = Weather_1,
                    name = 'Weather_Impact'

                ),
                Bar(
                    x = genre_names,
                    y = Weather_0,
                    name = 'Not_Weather_Impact'
                )
            ],

            'layout': {
                'title': 'Impact reported related to Weather',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }
        },                
        {
            'data': [
                Bar(
                    x = category_columns,
                    y = category_values
                )
            ],

            'layout': {
                'title': 'Distribution of Message Columns',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
    

