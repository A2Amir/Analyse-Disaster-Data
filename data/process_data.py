#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation


# import libraries
import pandas as pd
import numpy as  np
import sqlite3
import argparse
import os



def load_data(f1_dir, f2_dir):
    '''

    Output:
        df (pandas Dataframe): load the datasets and merge them into one and return it.
    '''
   # load messages dataset
    messages = pd.read_csv(f1_dir)

    # load categories dataset
    categories = pd.read_csv(f2_dir)
    
    # Looking at the shapes of the DataFrames:
    print('Rows and columns in disaster messages :', messages.shape)
    print('Rows and columns in disaster categories :', categories.shape)


    # - Merge the messages and categories datasets using the common id
    df = messages.merge(categories, on='id')

    # Looking at the shapes of the DataFrame:
    print('Rows and columns in the merged dataset:', df.shape)

    return df



def clean_data(df):
    
    '''
    Input:
        df(pandas Dataframe): dataset combining messages and categories
    Output:
        df(pandas Dataframe): Cleaned dataset
    '''
    # create a dataframe of the 36 individual category columns
    categories =  df.categories.str.split(pat=';', expand=True)


    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split('-')[0])


    # rename the columns of `categories`
    categories.columns = category_colnames


    # Convert category values to just numbers 0 or 1.
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1] if int(x.split('-')[1]) < 2 else 1)   

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])



    # Replace `categories` column in `df` with new category columns.

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join='inner', axis=1)


    # Remove duplicates
    df.drop_duplicates( inplace=True)
        
    return df


def save_data(df,output_dir):
    '''
    Input:
        df(pandas Dataframe): Cleaned dataset
    Output:
        df(pandas Dataframe): Save the clean dataset into an sqlite database
    '''
    #  Save the clean dataset into an sqlite database.
    engine = sqlite3.connect(os.path.join(path, output_dir))
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')

if __name__ == "__main__":

    path = os.getcwd()


    parser = argparse.ArgumentParser(description='prepration')
    parser.add_argument('--f1', help='The file address and neme of the disaster_messages csv file.')

    parser.add_argument('--f2', help='The file address and neme of the disaster_categories csv file.')

    parser.add_argument('--o',  help='The address and name of the sqLite database into which data must be saved.')


    args = parser.parse_args()

    if not args.f1:
        raise ImportError('The --f1 parameter needs to be provided (The file address and neme of the disaster_messages csv file)')
    else:
        f1_dir = os.path.join(path, args.f1)

    if not args.f2:
        raise ImportError('The --f2 parameter needs to be provided (The file address and neme of the disaster_categories csv file')
    else:
        f2_dir = os.path.join(path, args.f2)

    if not args.o:
        raise ImportError('The --o parameter needs to be provided (The address and name of the sqLite database into which data must be saved)')
    else:
        output_dir = os.path.join(path, args.o)

    

    df = load_data(f1_dir, f2_dir)
    clean_df = clean_data(df)
    save_data(clean_df,output_dir)
    print('is done...')




