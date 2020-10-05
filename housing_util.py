import os
import tarfile
import urllib.request
import pandas as pd
from zlib import crc32
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/DatalogiForAlle/Hus-Priser-og-Machine-Learning/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data():
    os.makedirs(HOUSING_PATH, exist_ok=True)
    tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")
    urllib.request.urlretrieve(HOUSING_URL, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=HOUSING_PATH)
    housing_tgz.close()


def prepare_data(data):
    data = data.drop("ocean_proximity", axis = 1)
    data = data.fillna(data.median()) # bruger ikke imputer da det ville resultere i at pandas dataframen bliver til et numpy array
    return data

def load_housing_data():
    csv_path = os.path.join(HOUSING_PATH, "housing.csv")
    housing = pd.read_csv(csv_path)
    return prepare_data(housing)

def download_housing_data():
    fetch_housing_data()
    return load_housing_data()

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

       


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

from sklearn.metrics import mean_squared_error
def CalculateAccuracy(testdata, labels, model):
    predictions = model.predict(testdata)
    mse = mean_squared_error(labels, predictions)
    return mse


from sklearn.model_selection import cross_val_score
def cross_validate(model, prepared, labels):
    scores = cross_val_score(model, prepared, labels,
                             scoring="neg_mean_squared_error", cv = 10)
    scores = np.sqrt(-scores)
    return scores

def display_scores(scores):
     print("Scores:", scores)
     print("Mean:", scores.mean())
     print("Standard deviation:", scores.std())
