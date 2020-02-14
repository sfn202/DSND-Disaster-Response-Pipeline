# Disaster Response Pipeline Project

This project, was about Disaster Response Pipeline. The data was provided from figure eight that data set containing real messages that were sent during disaster events.


## The following are the files in the repository :

disaster_categories (1).csv -- the first data was used

disaster_messages (1).csv -- the second data was used

process_data (1).py

templates.tar.gz

run.py -- the app of the project



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python train_classifier.py DisasterResponse.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


