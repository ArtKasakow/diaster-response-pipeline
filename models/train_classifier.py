import sys
import nltk

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

import pandas as pd
import numpy as np
import re
import pickle

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, classification_report, fbeta_score, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin

url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

def load_data(database_filepath):
    """
    Input:
    Reading in database filepath
    
    Output:
    Returning three variables for further Analysis.
    X: dependent variable that contains all the text messages to analyze
    y: independent variable (1 or 0) that contains all the possible outcomes according to the analyized text     message.
    categories: categorizes the independent variables in a list.
    """
    
    # load data from database
    db_name = 'sqlite:///{}'.format(database_filepath)
    engine = create_engine(db_name)
    
    # reading the SQL datafile into a dataframe
    df = pd.read_sql('DisasterResponse', engine)
    
    # splitting into variables X, y and categories
    X = df['message'].values
    y = df.iloc[:, 4:]
    categories = list(df.columns[4:])
    
    return X, y, categories


def tokenize(text):
    """ 
    Tokenizing the text input.
    
    Output: 
    clean_tokens (List): list of tokenized words for ML algorithm
    """

    # removing urls 
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove Stop Words
    words = [w for w in tokens if w not in stopwords.words('english')]
    
    lemmatizer = WordNetLemmatizer()

    # Create a list of clean tokens
    clean = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    clean_tokens = [lemmatizer.lemmatize(w, pos='v').strip() for w in clean]
    
    return clean_tokens


def build_model():
    """
    Building a machine learning pipeline that processes text messages.
    Uses different NLP and ML models to detect and classify multiple outputs.
    
    Output:
    Pipeline containing different ML models.
    
    """
    
    # creating a multiple output classifier
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    Predicting the results based on the test data given.
    Comparing test results based on the test data with the real results.
    
    """

    # predict on test data
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    """
    Export your model as a pickle file.
    Saves trained model as pickle file to be loaded later.
    
    """
    
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


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