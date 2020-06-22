# 1. Introduction

During a disaster, typically we will get millions and millions of communications either direct or via social media right at the time when disaster response organizations have the least capacity to filter and then pull out the messages which are the most important. Machine learning is critical to helping different organizations understand which messages are relevant to them, and which messages to prioritize.  


In this repo, I am going to be analyzing thousands of real messages of disaster data from [Figure Eight](), which contains pre-labeled tweets and text messages from real-life disasters, to create a model for an API that classifies disaster messages.

# 2. Project Components

There are three components for this project: 

1. **ETL Pipeline:** First I will repair the data with an ETL pipeline that process messages and category data from CSV file and load them into SQLite database. In the Python script, **process_data.py**, you will find the data cleaning pipeline that:

    * Loads the messages and categories datasets
    * Merges the two datasets
    * Cleans the data
    * Stores it in a SQLite database

2. **ML Pipeline:** Then use a machine learning pipeline to raed data from the SQLite database to create and save a multi-output supervised learning model. In the Python script, **train_classifier.py**, you will find the machine learning pipeline that:

    * Loads data from the SQLite database
    * Splits the dataset into training and test sets
    * Builds a text processing and machine learning pipeline
    * Trains and tunes a model using GridSearchCV
    * Outputs results on the test set
    * Exports the final model as a pickle file

3. **Flask Web App:** I will create a web application, which use the trained model(the pickle file) to classify incoming messages where an emergency worker can input a new message and get classification results in several categories.

# 3. Structure 
Below you can find the file structure of the project:
<pre><code class="lang-txt">
      - app
      | - template
      | |- master.html  # main page of web app
      | |- go.html  # classification result page of web app
      |- run.py  # Flask file that runs app

      - data
      |- disaster_categories.csv  # data to process 
      |- disaster_messages.csv  # data to process
      |- process_data.py
      |- InsertDatabaseName.db   # database to save clean data to

      - models
      |- train_classifier.py
      |- classifier.pkl  # saved model 

      - README.md
</code></pre>
