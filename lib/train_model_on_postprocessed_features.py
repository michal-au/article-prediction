import numpy as np
import scipy.sparse
import os

# musi zustat, pouziva se tu eval()
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier


from .plot import log_result
from .features import feature_names_to_abbrev
from .utils import read_settings, load_csr_matrix

SETTINGS = read_settings()


def train_model(features, model_call, train_dataset_name, eval_dataset_name):

    print('reading datasets ...')
    print('- {}'.format(train_dataset_name))
    train_x, train_y = load_postprocessed_dataset(train_dataset_name, features)
    eval_x, eval_y = load_postprocessed_dataset(eval_dataset_name, features)
    print('... done')


    model = eval(model_call)
    print('fitting the model ...')
    model.fit(train_x, train_y)
    print('... done')

    score = model.score(eval_x, eval_y)
    train_score = model.score(train_x, train_y)
    print('SCORE: {}, (train: {})'.format(score, train_score))

    model_type = model_call.split('(')[0]
    log_result(
        model_type, feature_names_to_abbrev(features), model_call, train_dataset_name, eval_dataset_name,
        score, train_score,
    )


def load_postprocessed_dataset(dataset_name, features):
    y = np.load(os.path.join(SETTINGS.get('paths', 'dataFeaturesPennPostprocessed'), dataset_name, 'Y_article'))
    feature_matrices = []
    for feature_name in features:
        feature_matrices.append(load_csr_matrix(
            os.path.join(SETTINGS.get('paths', 'dataFeaturesPennPostprocessed'), dataset_name, feature_name + '.npz')
        ))
    return scipy.sparse.hstack(feature_matrices), y
