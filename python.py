import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from datetime import datetime

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
            # check feature_mapper so that all keys are present
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
    #print("mean:", mean)
    #print("stdev:", stdev)
    #print("num null:", num_null)
    rand_values = np.random.rand(num_null) * (2 * stdrange * stdev) + mean - stdrange * stdev
    #print(rand_values)
    temp_copy = df[feature].copy()
    temp_copy[np.isnan(temp_copy)] = rand_values
    df[feature] = temp_copy


df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

most_common_embarked = df['Embarked'].describe()['top']

fill_feature_constant(df, 'Fare', 0)
fill_feature_constant(df, 'Embarked', most_common_embarked)
fill_feature_rand(df, 'Age')

features_to_map = ["Embarked", "Sex"]
mapper = {'Embarked': {'C': 0, 'Q': 1, 'S': 2}, 'Sex': {'female': 0, 'male': 1}}
mapped = encode(df, features_to_map, [mapper[x] for x in features_to_map])

train_X = df.drop('Survived', axis=1)
train_y = df['Survived']

tree = RandomForestClassifier(150)
scores = cross_val_score(tree, train_X, train_y, cv=10)
print(scores)
print("mean:", np.mean(scores))

exit(0)
tree.fit(train_X, train_y)

#print(tree.score(train_X, train_y))

test_csv = pd.read_csv("test.csv")
test_pass_ids = test_csv[['PassengerId']]
test_csv = test_csv.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
fill_feature_constant(test_csv, 'Fare', 0)
fill_feature_constant(test_csv, 'Embarked', most_common_embarked)
fill_feature_rand(test_csv, 'Age')

encode(test_csv, features_to_map, [mapper[x] for x in features_to_map])

#print(test_csv.head(15))
test_predictions = tree.predict(test_csv)

#print(test_predictions)

generated_df = test_pass_ids[["PassengerId"]]
generated_df["Survived"] = test_predictions

#print(generated_df.head(15))

generated_df.to_csv("generated_submissions/generated_submission_" + str(datetime.now().strftime("%y-%m-%d %H-%M-%S")) + ".csv", index=False)
