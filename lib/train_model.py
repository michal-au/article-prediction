import numpy as np  # type: ignore
import pandas  # type: ignore
import matplotlib.pyplot as plot  # type: ignore
import gc
import os
import scipy
import csv
import datetime

from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.cross_validation import cross_val_score

from ..extract_features.list_to_binary_features import resolve_list_features, remove_nans_from_dicts, prepare_for_vectorizer, convert_bools
from .plot import log_result, export_errors, create_result_df, get_accuracy
from .features import feature_names_to_abbrev
from .utils import create_dir_for_file, read_settings
from ..extract_features.df_transform import convert_list_vector, convert_nominal_vector, get_feature_value_counts, LIST_FEATURES
from ..experiments.df_transform.embeddings import process_embeddings
from ..experiments.df_transform.scaler import scale
from ..experiments.df_transform.cutoffer import cutoff_feature, cutoff_test_feature


def preprocess_dataframe(df, features):
    X, Y = _separate_target_variable(df)
    features_to_drop = [f for f in X.columns if f not in features]
    X = X.drop(features_to_drop, 1)
    print X.columns

    X, updated_features = process_embeddings(X, features)
    X = convert_bools(X)

    X, dict_data = prepare_for_vectorizer(X)
    return X, Y, dict_data


def tmp2(train_df, test_df, features):
    """
    train_df.a_head_form = train_df.a_head_form.apply(lambda x: x.lower())
    test_df.a_head_form = test_df.a_head_form.apply(lambda x: x.lower())
    train_df.a_words_after_np = train_df.a_words_after_np.apply(lambda x: [i.lower() for i in x])
    test_df.a_words_after_np = test_df.a_words_after_np.apply(lambda x: [i.lower() for i in x])
    train_df.a_words_before_np = train_df.a_words_before_np.apply(lambda x: [i.lower() for i in x])
    test_df.a_words_before_np = test_df.a_words_before_np.apply(lambda x: [i.lower() for i in x])
    """
    train_X, train_Y, dict_train = preprocess_dataframe(train_df, features)
    test_X, test_Y, dict_test = preprocess_dataframe(test_df, features)
    print 'dict vectorizer ...'
    vec = DictVectorizer()
    train_vec_array = vec.fit_transform(dict_train)
    train_array = scipy.sparse.csr_matrix(train_X.values)
    test_vec_array = vec.transform(dict_test)
    test_array = scipy.sparse.csr_matrix(test_X.values)
    if train_array.shape[1]:
        train_X = scipy.sparse.hstack([train_array, train_vec_array])
    else:
        train_X = train_vec_array
    if test_array.shape[1]:
        test_X = scipy.sparse.hstack([test_array, test_vec_array])
    else:
        test_X = test_vec_array
    print '... done'

    #train_X, test_X = scale(train_X, test_X)

    return train_X, train_Y, test_X, test_Y


def train_model(features, model_call, train_df_path, test_df_path, only_console=False, cross_validation=False, path=None):
    print 'reading datasets ...'
    train_df = pandas.read_pickle(train_df_path)
    test_df = pandas.read_pickle(test_df_path)
    print '... done'

    train_X, train_Y, test_X, test_Y = tmp2(train_df, test_df, features)

    model = eval(model_call)
    if cross_validation:
        scores = cross_val_score(model, train_X, train_Y, cv=10)
        score = scores.mean()
        print "SCORE: %0.5f (+/- %0.5f)" % (score, scores.std() * 2)
    else:
        print 'fitting the model ...'
        model.fit(train_X, train_Y)
        print '... done'

        score = model.score(test_X, test_Y)
        train_score = model.score(train_X, train_Y)
        print 'SCORE: {}, (train: {})'.format(score, train_score)

    if not only_console:
        settings = read_settings()
        model_type = model_call.split('(')[0]
        Y_predicted = model.predict(test_X)

        create_df_path = settings.get('paths', 'dataFeatures')
        create_df_path = os.path.join(create_df_path, 'predicted', model_type + '.pkl')
        create_dir_for_file(create_df_path)
        create_result_df(Y_predicted, test_df, create_df_path)

        log_result(
            model_type, feature_names_to_abbrev(features), model_call, train_df_path, test_df_path,
            score, train_score,
            path=path
        )

        #error_path = os.path.join(settings.get('paths', 'logModelResults'), 'classification_errors', model_type)
        #create_dir_for_file(error_path)
        #export_errors(model.predict(test_X), test_Y, test_df, error_path)


def train_svm_model(features, train_df_path, test_df_path, only_console=False):
    model_call = 'svm.SVC(decision_function_shape="ovo", gamma=0.1, C=1)'
    print 'reading datasets ...'
    train_df = pandas.read_pickle(train_df_path)
    test_df = pandas.read_pickle(test_df_path)
    print '... done'

    feature_value_counts = get_feature_value_counts(train_df, 5, features)

    train_df = train_df.sample(n=5000, random_state=42)
    train_X, train_Y = _separate_target_variable(train_df)
    test_X, test_Y = _separate_target_variable(test_df)

    train_X = df_features_to_probs(train_X, features, feature_value_counts)
    test_X = df_features_to_probs(test_X, features, feature_value_counts)

    print "rows: {}".format(len(train_X))
    print "columns: {}".format(len(train_X.columns))

    print test_X[:10]
    print test_X.columns

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(train_X)
    train_X = imp.transform(train_X)
    test_X = imp.transform(test_X)

    model = eval(model_call)
    print 'fitting the model ...'
    model.fit(train_X, train_Y)
    print '... done'
    print model
    score = model.score(test_X, test_Y)
    train_score = model.score(train_X, train_Y)
    print 'SCORE: {}, (train: {})'.format(score, train_score)

    if not only_console:
        settings = read_settings()
        model_type = model_call.split('(')[0]
        Y_predicted = model.predict(test_X)

        create_df_path = settings.get('paths', 'dataFeatures')
        create_df_path = os.path.join(create_df_path, 'predicted', model_type + '.pkl')
        create_dir_for_file(create_df_path)
        create_result_df(Y_predicted, test_df, create_df_path)

        log_result(model_type, features, model_call, train_df_path, test_df_path, score)

        error_path = os.path.join(settings.get('paths', 'logModelResults'), 'classification_errors', model_type)
        create_dir_for_file(error_path)
        export_errors(model.predict(test_X), test_Y, test_df, error_path)


def df_features_to_probs(df, features, feature_value_counts, cutoff=5):
    list_feature_names = [f[0] for f in LIST_FEATURES]
    for feature_name in df.columns:
        if feature_name not in features:
            df = df.drop(feature_name, 1)
    for feature_name in features:
        if feature_name in list_feature_names:
            df = convert_list_vector(df, feature_name, feature_value_counts)
        else:
            df = convert_nominal_vector(df, feature_name, feature_value_counts)
    return df


def _separate_target_variable(df):
    # type(dataframe) -> dataframe, List[str]
    """
    separates the target feature from the rest of the data

    :param df: pandas dataframe
    :return: (pandas dataframe: features, target feature)
    """
    Y = df['Y_article'].values  # target feature as an array
    X = df.drop('Y_article', 1)
    assert len(df.columns) == len(X.columns) + 1  # sanity check
    assert len(Y) == len(X.values)  # sanity check
    return X, Y


def assert_nparrays_equal(a1, a2):
    a1_list = [a1[:, i].tolist() for i in xrange(a1.shape[1])]
    for i in xrange(a2.shape[1]):
        assert a2[:, i].tolist() in a1_list


def translate_feature_names_to_delete(feature_names):
    features = (
        'a_head_form',  # 0
        'a_head_number',  # 1
        'a_head_pos',
        'a_hypernyms',  # 3
        'a_non_article_det',  # 4
        'a_parent',
        'a_pos_after_head',  # 6
        'a_pos_before_head',  # 7
        'a_referent',
        'a_words_after_head',  # 9
        'a_words_after_np',  # 10
        'a_words_before_head',  # 11
        'a_words_before_np',  # 12
        'b_head_pos_simple',  # 13
        'b_head_proper',  # 14
        'b_is_postmodified',
        'b_postmodification_length',
        'b_postmodification_pp_of',
        'b_postmodification_type',  # 18
        'b_postmodification_type_specific',
        'b_predicate_form',
        'b_referent',  # 21
        'b_subject_predicate2',
        'b_subject_position',
        'b_parent',  # 24
        'b_object_form',  # 25
        'b_object_form_cutoff',
        'b_pp_object_form',  # 27
        'c_countability',
        'd_head_form_embeddings',  # 29
        'd_object_form_embeddings',
        'd_words_before_head_embeddings',
        'd_words_after_head_embeddings',
        'd_words_before_np_embeddings',
        'd_words_after_np_embeddings',
        'b_subject_predicate',
        'b_position_within_article',
        'c_countability',
        'd_head_form_embeddings',
        'd_object_form_embeddings',
        'd_words_before_head_embeddings',
        'd_words_after_head_embeddings',
        'd_words_before_np_embeddings',
        'd_words_after_np_embeddings',
        'c_countability_bnc',  # 44
    )
    try:
        return [str(features.index(f)) for f in feature_names]
    except ValueError as ex:
        for f in feature_names:
            if f not in features:
                print f
        raise ex
