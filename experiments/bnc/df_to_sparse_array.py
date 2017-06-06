import os
import numpy
import pandas
import scipy
from sklearn.feature_extraction import DictVectorizer
from cPickle import dump

from ...lib.train_model import _separate_target_variable, tmp2
from ...lib.utils import read_settings, save_csr_matrix, load_csr_matrix
from ..df_transform.embeddings import process_embeddings
from ...extract_features.list_to_binary_features import resolve_list_features, remove_nans_from_dicts, prepare_for_vectorizer, convert_bools
from .tune import features as FEATURES

STEP = 1000000
SETTINGS = read_settings()
feature_path = SETTINGS.get('paths', 'dataFeatures')

#TRAIN_NAME = 'bnc-train260000'
TRAIN_NAME = 'bnc-train2000000'
TEST_NAME = 'bnc-test271385'

train_df_path = os.path.join(SETTINGS.get('paths', 'dataFeatures'), TRAIN_NAME + '.pkl')
test_df_path = os.path.join(SETTINGS.get('paths', 'dataFeatures'), TEST_NAME + '.pkl')
mtrx_path = SETTINGS.get('paths', 'dataFeaturesSparseBnc')

def save_vectorizer(vec):
    print 'saving vectorizer ...'
    dump(vec, open(os.path.join(SETTINGS.get('paths', 'model'), 'vectorizer.pkl'), 'wb'))
    print '... done'

if __name__ == '__main__':
    vec = DictVectorizer()

    df_length = 1999995

    i = 0
    while i * STEP < df_length:
        idx_from = i * STEP
        idx_to = min(((i + 1) * STEP, df_length))
        f_name = 'bnc-train-{}-{}'.format(idx_from, idx_to)
        print 'iteration: {} ({})'.format(i+1, f_name)

        print "loading df..."
        df = pandas.read_pickle(train_df_path)
        features_to_drop = [f for f in df.columns if f not in FEATURES and f != 'Y_article']
        X, Y = _separate_target_variable(df[idx_from: idx_to])
        df = None
        numpy.save(os.path.join(mtrx_path, f_name + '-Y'), Y)

        X = X.drop(features_to_drop, 1)
        X, _ = process_embeddings(X, FEATURES)
        X = convert_bools(X)

        X, dict_data = prepare_for_vectorizer(X)
        if i == 0:
            train_vec_array = vec.fit_transform(dict_data)
            save_vectorizer(vec)
            raise Exception
        else:
            train_vec_array = vec.transform(dict_data)

        print "... saving"
        mtx = scipy.sparse.hstack([scipy.sparse.csr_matrix(X.values), train_vec_array], format='csr')
        save_csr_matrix(mtx, os.path.join(mtrx_path, f_name))

        i += 1

    # TEST DATA:::::::::::::
    df = pandas.read_pickle(test_df_path)
    features_to_drop = [f for f in df.columns if f not in FEATURES and f != 'Y_article']
    X, Y = _separate_target_variable(df)
    df = None
    numpy.save(os.path.join(mtrx_path, TEST_NAME + '-Y'), Y)
    X = X.drop(features_to_drop, 1)
    X, _ = process_embeddings(X, FEATURES)
    X = convert_bools(X)
    X, dict_data = prepare_for_vectorizer(X)
    test_vec_array = vec.transform(dict_data)

    print "... saving"
    mtx = scipy.sparse.hstack([scipy.sparse.csr_matrix(X.values), test_vec_array], format='csr')
    save_csr_matrix(mtx, os.path.join(mtrx_path, TEST_NAME))
