import os
from os.path import *
import configparser  # type: ignore
import logging
import pandas
import scipy
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV

from .Tree import Tree

__all__ = ('read_settings', 'create_dir_for_file', 'print_list_to_file', 'set_logging', 'read_pickle')


def read_settings():
    """read the .settings file"""
    settings_path = join(dirname(dirname(__file__)), '.settings')
    filename = settings_path
    settings = configparser.ConfigParser()
    settings.read(filename)
    return settings


def create_dir_for_file(f_path):
    """checks whether the path exists, if not new directory is created"""
    d = os.path.dirname(f_path)
    if d and not os.path.exists(d):
        os.makedirs(d)


def print_list_to_file(lines, path):
    """writes the list (i.e. 'lines') line by line to the file, overwriting
    the original file if it exists"""
    create_dir_for_file(path)
    with open(path, 'w+') as f:
        for l in lines:
            print>>f, l


def set_logging(f_name):
    path = read_settings().get('paths', 'log')
    path = join(path, f_name)
    create_dir_for_file(path)
    logging.basicConfig(filename=path, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging


def read_pickle(name):
    settings = read_settings()
    path = settings.get('paths', 'dataFeatures')
    fpath = os.path.join(path, name)
    if not os.path.exists(fpath):
        raise NameError('Nemuzu najit {} (cesta: {})\nk dispozici: {}'.format(name, fpath, os.listdir(path)))
    else:
        return pandas.read_pickle(fpath)


def read_trees_from_file(f_path):
    trees = []
    with open(f_path, 'r') as f:
        for l in f:
            if l:
                trees.append(Tree.from_string(l))
    return trees


def save_csr_matrix(matrix, path):
    np.savez(path, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr, shape=matrix.shape)


def load_csr_matrix(path):
    loader = np.load(path)
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def load_postprocessed_feature_data(features, dataset_name, sample=None, data_path=None):
    if not data_path:
        data_path = read_settings().get('paths', 'dataFeaturesPennPostprocessed')
    y = np.load(os.path.join(data_path, dataset_name, 'Y_article'))
    if sample:
        np.random.seed(seed=42)
        sample_indices = [np.random.choice(len(y), sample, replace=False)]
        y = y[sample_indices]

    feature_matrices = []
    for feature_name in features:
        feature_matrix = load_csr_matrix(
            os.path.join(data_path, dataset_name, feature_name + '.npz')
        )
        if sample:
            feature_matrix = scipy.sparse.csr_matrix(feature_matrix.toarray()[sample_indices])
        feature_matrices.append(feature_matrix)
    x = scipy.sparse.hstack(feature_matrices)
    assert x.shape[0] == len(y)
    return x, y


def logreg_gridsearch(x, y, log_file_path, model_params, tuning_params, cores=5):
    start_time = time.time()
    gsearch = GridSearchCV(
        estimator=LogisticRegression(**model_params),
        param_grid=tuning_params, n_jobs=cores, cv=5, verbose=10
    )
    gsearch.fit(x, y)
    pandas.DataFrame(gsearch.cv_results_).to_csv(log_file_path)
    print(gsearch.best_estimator_, gsearch.best_score_)
    end_time = time.time()
    print("Done in {} minutes (used cores: {})".format((end_time - start_time)/60, cores))