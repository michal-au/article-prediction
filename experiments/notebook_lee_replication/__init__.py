__all__ = (
    'pandas',
    'train_X_orig',
    'train_Y',
    'test_X_orig',
    'test_Y',
    'HTML',
    'resolve_list_features',
    'remove_nans_from_dicts',
    'DictVectorizer',
    'LogisticRegression',
    'confusion_matrix',
    'plot_cm',
    'export_errors',
    'log_path',
)


import os
import sys
root_dir = os.path.join(os.getcwd(), os.path.join('../..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)


from lib.utils import read_settings
settings = read_settings()
path = settings.get('paths', 'dataFeatures')
path = os.path.join(root_dir, '..', path)

log_path = settings.get('paths', 'log')
log_path = os.path.join('../../..', log_path, 'experiments/replication/classification_errors')


import pandas
df_train = pandas.read_pickle(os.path.join(path, "train.pkl"))
df_test = pandas.read_pickle(os.path.join(path, "test.pkl"))

train_Y = df_train['Y_article'].values  # target feature as an array
train_X_orig = df_train.drop('Y_article', 1)  # attributes as a dataframe
assert len(train_X_orig.columns) == len(df_train.columns) - 1  # sanity check

test_Y = df_test['Y_article'].values  # target feature as an array
test_X_orig = df_test.drop('Y_article', 1)  # attributes as a dataframe
assert len(test_X_orig.columns) == len(df_test.columns) - 1  # sanity check

from extract_features.list_to_binary_features import resolve_list_features, remove_nans_from_dicts
from IPython.display import HTML
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from lib.plot import plot_cm, export_errors
