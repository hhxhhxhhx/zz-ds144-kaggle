import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from datetime import datetime

USING_PCLASS = True
USING_PREFIX = False
USING_SEX = True
USING_AGE = True
USING_SIBSP = True
USING_PARCH = True
USING_FARE = True
USING_EMBARKED = True

df = pd.read_csv("train.csv")

def one_hot_encode(df, features):
    new_df = df.copy(deep=True)
    for feature in features:
        unique_vals = np.unique(df[feature].to_numpy())
        for val in unique_vals:
            new_df[feature + "_" + str(val)] = df[feature].apply(lambda x: int(x == val))
    return new_df.drop(features, axis=1)

def encode(df, features, feature_mapper=None):
    if not feature_mapper:
        if type(features) == list:
            mappers = {}
            for feature in features:
                unique_vals = np.unique(df[feature].to_numpy())
                mapper = {}
                for i in range(len(unique_vals)):
                    mapper[unique_vals[i]] = i
                df["Mapped " + str(feature)] = df[feature].map(mapper)
                mappers[feature] = mapper
            df.drop(features, axis=1, inplace=True)
            return mappers
        else:
            unique_vals = np.unique(df[features].to_numpy())
            mapper = {}
            for i in range(len(unique_vals)):
                mapper[unique_vals[i]] = i
            df["Mapped " + str(features)] = df[features].map(mapper)
            df.drop(features, axis=1, inplace=True)
            return mapper
    else:
        if type(features) == list:
            if type(feature_mapper) != list:
                print("Feature mapper must be a list when features is a list of features!")
                return
            for i in range(len(features)):
                feature = features[i]
                encode_temp_copy = df[feature]
                unique_vals = np.unique(encode_temp_copy.to_numpy())
                for val in unique_vals:
                    if val not in feature_mapper[i]:
                        print("%s is present in df[%s], but not in feature_mapper!", val, feature)
                        return
                for key in feature_mapper[i]:
                    if key not in unique_vals:
                        print("%s is present in feature_mapper, but not in df[%s]!", key, feature)
                        return
                encode_temp_copy = encode_temp_copy.map(feature_mapper[i])
                df["Mapped " + str(feature)] = encode_temp_copy
            df.drop(features, axis=1, inplace=True)

        else:
            if type(feature_mapper) != dict:
                print("Feature mapper is not a dictionary when \"features\" is a single string!")
                return
            encode_temp_copy = df[features]
            unique_vals = np.unique(encode_temp_copy.to_numpy())
            for val in unique_vals:
                if val not in feature_mapper:
                    print("%s is present in df[%s], but not in feature_mapper!", val, features)
                    return
            for key in feature_mapper:
                if key not in unique_vals:
                    print("%s is present in feature_mapper, but not in df[%s]!", key, features)
                    return
            encode_temp_copy = encode_temp_copy.map(feature_mapper)
            df["Mapped " + str(features)] = encode_temp_copy
            df.drop(features, axis=1, inplace=True)

def fill_feature_constant(df, feature, constant):
    df[feature] = df[feature].fillna(constant)

def fill_feature_rand(df, feature, stdrange=1.5):
    mean = df[feature].mean()
    stdev = df[feature].std()
    num_null = df[feature].isnull().sum()
    rand_values = np.random.rand(num_null) * (2 * stdrange * stdev) + mean - stdrange * stdev
    temp_copy = df[feature].copy()
    temp_copy[np.isnan(temp_copy)] = rand_values
    df[feature] = temp_copy

popular_prefix = ["Mr.", "Mrs.", "Miss.", "Master."]

def get_prefix(name):
    prefix = re.search(r'[a-zA-Z]+, ([a-zA-Z]+.)', name, re.IGNORECASE).group(1)
    if prefix in popular_prefix:
        return prefix
    else:
        return "Other"

most_common_embarked = df['Embarked'].describe()['top']

features_to_map = ["Embarked", "Sex", "Prefix"]
mapper = {'Embarked': {'C': 0, 'Q': 1, 'S': 2}, 'Sex': {'female': 0, 'male': 1}, 'Prefix': {'Mr.': 0, 'Mrs.': 1, 'Miss.': 2, 'Master.': 3, 'Other': 4}}

def process_df(df):

    df = df.drop(["PassengerId", "Ticket", "Cabin"], axis=1)

    if not USING_PCLASS:
        df = df.drop("Pclass", axis=1)
    if not USING_PREFIX:
        df = df.drop("Name", axis=1)
    if not USING_SEX:
        df = df.drop("Sex", axis=1)
    if not USING_AGE:
        df = df.drop("Age", axis=1)
    if not USING_SIBSP:
        df = df.drop("SibSp", axis=1)
    if not USING_PARCH:
        df = df.drop("Parch", axis=1)
    if not USING_FARE:
        df = df.drop("Fare", axis=1)
    if not USING_EMBARKED:
        df = df.drop("Embarked", axis=1)

    if USING_PREFIX:
        df["Prefix"] = df["Name"].apply(get_prefix)
        df = df.drop("Name", axis=1)

    if USING_FARE:
        fill_feature_constant(df, "Fare", 0)
    if USING_EMBARKED:
        fill_feature_constant(df, "Embarked", most_common_embarked)
    if USING_AGE:
        fill_feature_rand(df, "Age")

    list_builder = []
    if USING_EMBARKED:
        list_builder.append("Embarked")
    if USING_SEX:
        list_builder.append("Sex")
    if USING_PREFIX:
        list_builder.append("Prefix")

    encode(df, list_builder, [mapper[x] for x in list_builder])

    return df

mapped = process_df(df)

train_X = mapped.drop('Survived', axis=1)
train_y = mapped['Survived']

params = {"n_estimators": [120, 130, 140, 150, 160, 170, 180, 190], 
          "criterion": ["gini", "entropy"], 
          "min_samples_split": [0.1, 0.2, 0.3, 0.4, 0.5], 
          "min_samples_leaf": [0.1, 0.2, 0.3, 0.4, 0.5]}

tree = RandomForestClassifier(140, min_samples_split=0.2, min_samples_leaf=0.2, criterion="entropy")

def print_scores(cv=10, iterations=1):
    means = []
    for _ in range(iterations):
        scores = cross_val_score(tree, train_X, train_y, cv=cv)
        means.append(np.mean(scores))
    print("Means:", means)
    print("Means of means:", np.mean(means))
    exit(0)

def do_random_search_cv(parameters, n_iter=200, cv=5):
    dat = RandomizedSearchCV(tree, parameters, n_iter=n_iter, cv=cv)
    dat.fit(train_X, train_y)
    print("Best parameters:", dat.best_params_)
    exit(0)

def do_grid_search_cv(parameters, cv=5):
    dat = GridSearchCV(tree, parameters, cv=cv)
    dat.fit(train_X, train_y)
    print("Best parameters:", dat.best_params_)
    exit(0)


# print_scores(iterations=5) # uncomment this line to print cross validation scores

# do_random_search_cv(params) # uncomment this line to print best hyperparameters using RandomSearchCV

# do_grid_search_cv(params) # uncomment this line to print best hyperparameters using GridSearchCV

# otherwise, fit this and output the predictions into a "generated_submissions/" directory

tree.fit(train_X, train_y)

test_csv = pd.read_csv("test.csv")
test_pass_ids = test_csv[['PassengerId']]

processed_test_csv = process_df(test_csv)

test_predictions = tree.predict(processed_test_csv)

generated_df = test_pass_ids[["PassengerId"]]
generated_df["Survived"] = test_predictions

import os

if not os.path.isdir("generated_submissions"):
    os.makedirs("generated_submissions")
    print("Created directory generated_submissions")

datetime_str = str(datetime.now().strftime("%y-%m-%d %H-%M-%S"))

generated_df.to_csv("generated_submissions/generated_submission_" + datetime_str + ".csv", index=False)

parameters_used = {}
parameters_used["n_estimators"] = tree.n_estimators
parameters_used["criterion"] = tree.criterion
parameters_used["min_samples_split"] = tree.min_samples_split
parameters_used["min_samples_leaf"] = tree.min_samples_leaf

import json

if not os.path.isdir("classifier_settings"):
    os.makedirs("classifier_settings")
    print("Created directory classifier_settings")

with open("classifier_settings/setting_" + datetime_str + ".json", "w") as f:
    json.dump(parameters_used, f)
