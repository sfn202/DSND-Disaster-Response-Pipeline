import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    disaster_messages = pd.read_csv(messages_filepath)
    disaster_categories = pd.read_csv(categories_filepath)
    df = pd.concat([disaster_messages, disaster_categories])
    return df


def clean_data(df):
    
    categories = df['categories'].str.split(';', n=36, expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split('-').apply(lambda x:x[0])
    categories.columns = category_colnames
    for i in categories:
       # cate = categories.iloc[1]
        #cate[i] = cate[i].astype(str).split('-').apply(lambda x:x[1])
        categories[i] = categories[i].astype(str).str[-1]
        categories[i] = cate[i].astype(int)
        #df = df.drop(columns=['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    df = df.drop(columns=['categories'])
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    
    df.to_sql('disaster_response', engine, index=False)
    return df



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
        


if __name__ == '__main__':
    main()