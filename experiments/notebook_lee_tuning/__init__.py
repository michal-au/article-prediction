__all__ = (
    'pandas',
    'train_X_orig',
    'train_Y',
    'heldout_X_orig',
    'heldout_Y',
    'train_tun_X_orig',
    'train_tun_Y',
    'heldout_tun_X_orig',
    'heldout_tun_Y',
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
df_heldout = pandas.read_pickle(os.path.join(path, "heldout.pkl"))
df_train_tun = pandas.read_pickle(os.path.join(path, "tun_train.pkl"))
df_heldout_tun = pandas.read_pickle(os.path.join(path, "tun_heldout.pkl"))


train_Y = df_train['Y_article'].values  # target feature as an array
train_X_orig = df_train.drop('Y_article', 1)  # attributes as a dataframe
assert len(train_X_orig.columns) == len(df_train.columns) - 1  # sanity check

heldout_Y = df_heldout['Y_article'].values  # target feature as an array
heldout_X_orig = df_heldout.drop('Y_article', 1)  # attributes as a dataframe
assert len(heldout_X_orig.columns) == len(df_heldout.columns) - 1  # sanity check

train_tun_Y = df_train_tun['Y_article'].values  # target feature as an array
train_tun_X_orig = df_train_tun.drop('Y_article', 1)  # attributes as a dataframe
assert len(train_tun_X_orig.columns) == len(df_train_tun.columns) - 1  # sanity check

heldout_tun_Y = df_heldout_tun['Y_article'].values  # target feature as an array
heldout_tun_X_orig = df_heldout_tun.drop('Y_article', 1)  # attributes as a dataframe
assert len(heldout_tun_X_orig.columns) == len(df_heldout_tun.columns) - 1  # sanity check


from extract_features.list_to_binary_features import resolve_list_features, remove_nans_from_dicts
from IPython.display import HTML
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from lib.plot import plot_cm, export_errors
