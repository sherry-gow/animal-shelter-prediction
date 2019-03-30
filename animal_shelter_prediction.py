#!/usr/bin/env python
# coding: utf-8

# ## Exercise through the following article from Medium
# __[predicting Animal Selter Outcomes ] (https://medium.com/vickdata/predicting-animal-shelter-outcomes-4c5fad5dbb4f)__
#


import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')



#creating age_numeric feature: based on the AgeuponOutcome column, then drop AgeuponOutcome
def age_converter(row):
    age_string = row['AgeuponOutcome']
    [age,unit] = age_string.split(" ")
    unit = unit.lower()
    if("day" in unit):
        if age=='0': return 1
        return int(age)
    if("week" in unit):
        if(age)=='0': return 7
        return int(age)*7
    elif("month" in unit):
        if(age)=='0': return 30
        return int(age) * 30
    elif("year" in unit):
        if(age)=='0': return 365
        return int(age) *365

def prepare_features():
    train=pd.read_csv('data/train.csv')
    print('++++++++++++++++++++++++++++')
    print('train shape: {0}'.format(train.shape))
    print('++++++++++++++++++++++++++++')
    print(train.head())
    train['OutcomeType'].value_counts().plot.bar()
    plt.savefig('OutcomeType')
    #print('++++++++++++++++++++++++++++')
    #find out the data type to determine categorical or numeric features
    #print('train.dttypes: {0}'.format(train.dtypes)

    #find out the unique values of each column for feature preparation
    columns = train.columns
    for column in columns:
        print(column)
        print(train[column].nunique())


    # Find out the pct of the missing data
    train.apply(lambda x: sum(x.isnull()/len(train)))
    #drop OutcomeSubtype column as missing pct is high
    train = train.drop('OutcomeSubtype', axis=1)
    # ### Deal with Missing Data for Name column, conver it to has_name feature,
    #     then drop Name column
    train['Name'] = train[['Name']].fillna(value=0)
    train['has_name'] = (train['Name'] != 0).astype('int64')
    train = train.drop('Name', axis=1)

    #find unique values of the Color column, sort descending order
    print( train['Color'].value_counts().index[0])
    #fill missing values for the remaining columns: SexuponOutcome and AgeuponOutcome
    train = train.apply(lambda x : x.fillna(x.value_counts().index[0])  )
    train.apply(lambda x: sum(x.isnull()/len(train)))

    #drop AnimalID column
    train = train.drop('AnimalID', axis = 1)

    #handle Color feature: it has high cardinality, so fill the count<300 with the value 'Others'
    color_counts = train['Color'].value_counts()
    color_others = set(color_counts[color_counts<300].index )
    train['top_colors'] = train['Color'].replace(list(color_others), 'Others')
    print(train['top_colors'].nunique())


    #handle Breed feature: fill value with no keyword Mix for 'pure', create 'breed_type'
    train['breed_type'] = train['Breed'].str.extract('({})'.format('|'.join(['Mix'])), flags=re.IGNORECASE, expand=False).str.lower().fillna('pure')

    #create multi_color feature: if colors column has '/' then set to 1 else 0
    train['multi_color']  = train['Color'].apply(lambda x: 1 if '/' in x else 0)
    train.head()

    train['age_numeric'] = train.apply(age_converter, axis=1)
    train = train.drop('AgeuponOutcome', axis=1)


    train = train.drop(['Breed', 'Color', 'DateTime'], axis=1)
    numeric_features = train.select_dtypes(include=['int64','float64']).columns
    # the following columns are categorical features:
    # Index(['AnimalType', 'SexuponOutcome', 'top_colors', 'breed_type'], dtype='object')
    categorical_features = train.select_dtypes(include = ['object']).drop(['OutcomeType'], axis=1).columns
    # Use Pandas.get_dummies() method to create one-hot encoding for those fields
    dummy_columns = pd.get_dummies(train[categorical_features])

    #find cleanup for the features
    final_train = pd.concat([dummy_columns, train], axis=1)
    final_train = final_train.drop(['AnimalType', 'breed_type', 'SexuponOutcome', 'top_colors'], axis=1)
    return final_train


def model_training_predicting(final_train):


    X = final_train.drop('OutcomeType', axis = 1)
    y = final_train['OutcomeType']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    rf_model = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    y_prob = rf_model.predict_proba(X_test)
    print(log_loss(y_test, y_prob))


    features = X.columns
    importances = rf_model.feature_importances_
    indices =np.argsort(importances)

    plt.figure(figsize=(10,20))
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.savefig('feature importance')


if __name__ == '__main__':
    final_train = prepare_features()
    model_training_predicting(final_train)
