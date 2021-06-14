# Disaster Response Pipeline

---

### Table of Contents

- [Description](#description)
- [How To Use](#how-to-use)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Author Info](#author-info)

---

## Description

This project will analyze thousands of real messages that were sent during natural disaster. These messages are collected from either social media or directly to disaster
response organizations. Therefore this project will build a ETL pipeline that processes message and category data and loading it in a SQLite Database. Afterwards the processed data will be analyzed using a ML pipeline to predict a multi-output supervised learning model. In the final stage the app will the deployed on the web to provide data visualizations and using the models to classify new messages for the 36 different categories to prioritize which message is important and relevant to a natural disaster.

With the help of Figure Eight, who provide the data, this project contributes to help and filter out messages that matter and finding basic messages by using keyword searches to provide trivial results.

#### Technologies

- Python
- Libraries:
    - Pandas
    - Numpy
    - Scikit-learn
    - NLTK
    - Flask
    - Plotly
- SQLalchemy

---

## How To Use

#### Installation

First clone this GIT repository:

`git clone https://github.com/ArtKasakow/disaster-response-pipeline`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

---

## License

All licenses in this repository are copyrighted by their respective authors.

Everything else is released under CC0.

------------------------------------------------------------------------------

No Copyright

The person who associated a work with this deed has dedicated the work to the
public domain by waiving all of his or her rights to the work worldwide under
copyright law, including all related and neighboring rights,
to the extent allowed by law.

You can copy, modify, distribute and perform the work, even for commercial
purposes, all without asking permission. See Other Information below.

Other Information:

    * In no way are the patent or trademark rights of any person affected
    by CC0, nor are the rights that other persons may have in the work or in
    how the work is used, such as publicity or privacy rights.

    * Unless expressly stated otherwise, the person who associated a work with
    this deed makes no warranties about the work, and disclaims liability for
    all uses of the work, to the fullest extent permitted by applicable law.

    * When using or citing the work, you should not imply endorsement
    by the author or the affirmer.


---

## Acknowledgements

I'm thanking Udacity and Figure Eight for this cooperation to provide a real dataset and helping to provide meaningful work and value.
Engineering a project from the start to help people that affected by natural disasters around the work and helping them in the best possible way to overcome natural disasters.

---

## Author Info

- Twitter - [@ArturKasakow](https://twitter.com/arturkasakow)
- LinkedIn - [Artur Kasakow](https://linkedin.com/in/arturkasakow/)
