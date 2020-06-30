from disaster_app import app
import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import os 
import sqlite3
from joblib import load 
import glob
import sys  
sys.path.append('models/')  
from utils import * #scriptName without .py extension  




# load data
database_path =  glob.glob('data/*.db') [0]
print('Database Path: ', database_path)
df, X, y, category_names = load_data(database_path)






# load model
pickle_path =  glob.glob('models/*.pkl') 
print(pickle_path)

#print('build model Path: ',pickle_path[0])
#print('get fbeta score Path: ',pickle_path[1])
print('Model Path: ',pickle_path[0])
model =load(pickle_path[0])   




   #f = open(model_path, 'rb')
#build_model =load(open(pickle_path[0], 'rb'))   
#get_fbeta_score = load(open(pickle_path[1], 'rb')) 
#model =load(open(pickle_path[2], 'rb'))   
#pipline = load(open(pickle_path[3], 'rb')) 

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    ### data for visualizing category counts.
    label_sums = df.iloc[:, 4:].sum()
    label_names = list(label_sums.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
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
                    x=label_names,
                    y=label_sums,
                )
            ],

            'layout': {
                'title': 'Distribution of labels/categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {

                },
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
#if __name__ == '__main__':
#    app.run(port=5000, debug=True)