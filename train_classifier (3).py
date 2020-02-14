import sys

import nltk

nltk.download('punkt')

nltk.download('wordnet')

nltk.download('stopwords')

nltk.download('averaged_perceptron_tagger')

import numpy as np

import pandas as pd

from sqlalchemy import create_engine

from nltk import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib


def load_data(database_filepath):
    create = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response',create)
    meassge1 = df['message']
    whole = df.drop(columns=['id','message','original','genre'])
    
    return message1, whole


def tokenize(text):
    v1 = word_tokenize(text)
    v2 = WordNetLemmatizer()
    v3 = set(v3.words('english'))
    clean = []
    for i in v1:
        clean = v2.lemmatize(i).lower().strip()
        if i not in v3:
            clean.append(cleaned)

    return cleaned


def build_model():
    p = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('text_len', TextLengthExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])
    # Set up the search grid
    parameters = {
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100]
    }
    # Initialize GridSearch cross validation object
    GridSearch = GridSearchCV(p, param_grid=parameters,n_jobs=-1)

    return GridSearch



def evaluate_model(model, X1, Y1, category_names):
     Y2= model.predict(X1)
    # Turn prediction into DataFrame
    Y2 = pd.DataFrame(Y2,columns=category_names)
    # For each category column, print performance
    for i in category_names:
        print(f'Column Name:{i}\n')
        print(classification_report(Y1[i],Y2[i]))
        



def save_model(model, model_filepath):
    joblib.dump(model, model_filepath) 



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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