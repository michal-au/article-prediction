from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from code.lib.utils import load_postprocessed_feature_data
import time
import os
import pandas as pd
import scipy
import numpy as np
from code.lib.utils import read_settings
import pickle
from code.experiments.logreg_multimodels.pairwise_coupling import recount_probs_by_pwc

FEATURES = ['a_hypernyms', 'a_head_form', 'a_head_number', 'a_non_article_det', 'a_parent', 'a_pos_after_head', 'a_pos_before_head', 'a_words_after_head', 'a_words_after_np', 'a_words_before_head', 'a_words_before_np', 'b_head_proper', 'b_head_pos_simple', 'b_object_form', 'b_pos_after_head_as_list', 'b_pos_before_head_as_list', 'b_pp_object_form', 'b_postmodification_type', 'b_referent', 'b_words_after_head_as_list', 'b_words_after_np_as_list', 'b_words_before_head_as_list', 'b_words_before_np_as_list', 'c_countability_bnc', 'd_head_form_embeddings', 'e_kenlm_ggl_5_lc_nbs']
TRAIN_SET_NAME = 'train'
TEST_SET_NAME = 'test'
TRAINING_DATA_SIZE = 263088  # 20000 | 50000 | 100000 | 263088
SETTINGS = read_settings()
CV_RESULTS_LOG_PATH = os.path.join(SETTINGS.get('paths', 'logModelResults'), 'penn', 'logreg_multi')
MODEL_PATH = SETTINGS.get('paths', 'model')


def _gridsearch(x, y, log_file_name, tuning_params, model_params, cores=5):
    start_time = time.time()
    gsearch = GridSearchCV(
        estimator=LogisticRegression(**model_params),
        param_grid=tuning_params, n_jobs=cores, cv=5, verbose=10
    )
    gsearch.fit(x, y)
    pd.DataFrame(gsearch.grid_scores_).to_csv(os.path.join(CV_RESULTS_LOG_PATH, log_file_name))
    print(gsearch.best_estimator_, gsearch.best_score_)
    end_time = time.time()
    print("Done in {} minutes (used cores: {})".format((end_time - start_time)/60, cores))


def _prepare_training_data(data, pair):
    # type: (dict, tuple) -> tuple
    sample_indices = np.where((data['train_y'] == pair[0]) | (data['train_y'] == pair[1]))
    return (
        scipy.sparse.csr_matrix(data['train_x'].toarray()[sample_indices]),
        data['train_y'][sample_indices],
    )


def _prepare_ovr_training_data(data, target):
    train_y = np.copy(data['train_y'])
    train_y[np.where(train_y != target)] = 'non-' + target
    assert len(train_y) == data['train_x'].shape[0]
    assert len(np.unique(train_y)) == 2, np.unique(train_y)
    print(np.unique(train_y))
    return data['train_x'], train_y


def _save_model(model, name):
    pickle.dump(model, open(os.path.join(MODEL_PATH, name), 'wb'))


def _load_model(name):
    return pickle.load(open(os.path.join(MODEL_PATH, name), 'rb'))


def multiclass_baseline_gridsearch(data):
    #=> 0.3 vychazi jako nejlepci
    model_params = {
        'multi_class': 'multinomial',
        'random_state': 42,
    }
    tuning_params = {
        'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
    }
    _gridsearch(
        data['train_x'],
        data['train_y'],
        'multiclass_cv_{}.csv'.format(TRAINING_DATA_SIZE),
        tuning_params,
        model_params
    )


def binary_gridsearch(data):
    model_params = {
        'random_state': 42,
    }
    tuning_params = {
        'C': [1/0.9, 1/0.8, 1/0.7, 1/0.6, 1/0.5], #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'penalty': ['l2'], #['l1', 'l2'],
        'solver': ['liblinear'],
    }
    for pair in (('THE', 'A'), ('THE', 'ZERO'), ('A', 'ZERO')):
        model_name = '{}-{}'.format(*pair)
        print("Grid Search for {} -------------------------------------------------".format(model_name))
        _gridsearch(
            *_prepare_training_data(data, pair),
            'binary_cv_{}_{}__.csv'.format(model_name, TRAINING_DATA_SIZE),
            tuning_params,
            model_params
        )


def binary_ovr_gridsearch(data):
    model_params = {
        'random_state': 42,
    }
    tuning_params = {
        'C': [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1/0.9], #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1/0.9, 1/0.8, 1/0.7, 1/0.6, 1/0.5],
        'penalty': ['l2'], #['l1', 'l2'],
        'solver': ['liblinear'],
    }
    for target in ('THE', 'A', 'ZERO'):
        print("Grid Search for {} -------------------------------------------------".format(target))
        _gridsearch(
            *_prepare_ovr_training_data(data, target),
            'ovr_binary_cv_{}_{}.csv'.format(target, TRAINING_DATA_SIZE),
            tuning_params,
            model_params
        )


def binary_ovr_outofbox_gridsearch(data):
    model_params = {
        'multi_class': 'ovr',
        'random_state': 42,
    }
    tuning_params = {
        'C': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1/0.9, 1/0.8, 1/0.7, 1/0.6, 1/0.5],
        'penalty': ['l2'],
        'solver': ['liblinear'],
    }
    _gridsearch(
        data['train_x'], data['train_y'],
        'ovr_outofbox_binary_cv_{}.csv'.format(TRAINING_DATA_SIZE),
        tuning_params,
        model_params
    )


def train_multiclass_baseline_model(data):
    model_params = {
        'multi_class': 'multinomial',
        'random_state': 42,
        'C': 0.3,
        'penalty': 'l2',
        'solver': 'lbfgs',
    }
    classifier = LogisticRegression(**model_params)
    classifier.fit(data['train_x'], data['train_y'])
    _save_model(classifier, 'logreg_multiclass_baseline_allfeatures_{}.pkl'.format(TRAINING_DATA_SIZE))


def train_binary_models(data):
    DATA_SIZE_TO_REGULARIZATION_PARAM_MAP = {  # vysledek krosvalidace
        'A-ZERO': {20000: 0.9, 50000: 1, 100000: 0.9, 263088: 1.25},
        'THE-A': {20000: 0.2, 50000: 0.3, 100000: 0.5, 263088: 0.4},
        'THE-ZERO': {20000: 0.9, 50000: 0.5, 100000: 0.8, 263088: 0.7},
    }
    model_params = {
        'random_state': 42,
        'solver': 'liblinear',
        'penalty': 'l2'  # vysledek krosvalidace, vzdycky nejlepsi
    }
    for pair in (('THE', 'A'), ('THE', 'ZERO'), ('A', 'ZERO')):
        model_name = '{}-{}'.format(*pair)
        print("Training {} -------------------------------------------------".format(model_name))
        classifier = LogisticRegression(C=DATA_SIZE_TO_REGULARIZATION_PARAM_MAP[model_name][TRAINING_DATA_SIZE], **model_params)
        classifier.fit(*_prepare_training_data(data, pair))
        _save_model(classifier, 'logreg_binarized_{}_allfeatures_{}.pkl'.format(model_name, TRAINING_DATA_SIZE))


def train_ovr_binary_models(data):
    DATA_SIZE_TO_REGULARIZATION_PARAM_MAP = {  # vysledek krosvalidace
        'A': {20000: 0.7, 50000: 0.7, 100000: 0.7, 263088: 0.6},
        'THE': {20000: 0.6, 50000: 0.6, 100000: 0.8, 263088: 0.5},
        'ZERO': {20000: 1, 50000: 0.8, 100000: 0.7, 263088: 0.7},
    }
    model_params = {
        'random_state': 42,
        'solver': 'liblinear',
        'penalty': 'l2'  # vysledek krosvalidace, vzdycky nejlepsi
    }
    for target in ('THE', 'A', 'ZERO'):
        print("Training {} -------------------------------------------------".format(target))
        classifier = LogisticRegression(C=DATA_SIZE_TO_REGULARIZATION_PARAM_MAP[target][TRAINING_DATA_SIZE], **model_params)
        classifier.fit(*_prepare_ovr_training_data(data, target))
        _save_model(classifier, 'logreg_ovr_binarized_{}_allfeatures_{}.pkl'.format(target, TRAINING_DATA_SIZE))


def train_ovr_outofbox_binary_models(data):
    DATA_SIZE_TO_REGULARIZATION_PARAM_MAP = {  # vysledek krosvalidace
        20000: 0.4, 50000: 0.7, 100000: 0.6, 263088: 0.6,
    }
    model_params = {
        'multi_class': 'ovr',
        'random_state': 42,
        'solver': 'liblinear',
        'C': DATA_SIZE_TO_REGULARIZATION_PARAM_MAP[TRAINING_DATA_SIZE],
        'penalty': 'l2'  # vysledek krosvalidace, vzdycky nejlepsi
    }
    classifier = LogisticRegression(**model_params)
    classifier.fit(data['train_x'], data['train_y'])
    _save_model(classifier, 'logreg_ovr_outofbox_binarized_allfeatures_{}.pkl'.format(TRAINING_DATA_SIZE))


def eval_multiclass_baseline_model(data):
    model_name = 'logreg_multiclass_baseline_allfeatures_{}'.format(TRAINING_DATA_SIZE)
    classifier = _load_model(model_name + '.pkl')
    train_predictions = classifier.predict(data['train_x'])
    test_predictions = classifier.predict(data['test_x'])
    with open(os.path.join(CV_RESULTS_LOG_PATH, model_name + '.txt'), 'a+') as f:
        f.write(str(classifier.get_params()) + '\n')
        f.write("Accuracy ({}): {:.4f}\n".format(TRAIN_SET_NAME, metrics.accuracy_score(data['train_y'], train_predictions)))
        f.write("Accuracy ({}): {:.4f}\n".format(TEST_SET_NAME, metrics.accuracy_score(data['test_y'], test_predictions)))


def eval_binary_models_majvote(data):
    test_predictions = {}
    train_predictions = {}
    for pair in (('THE', 'A'), ('THE', 'ZERO'), ('A', 'ZERO')):
        model_name = '{}-{}'.format(*pair)
        m = _load_model('logreg_binarized_{}_allfeatures_{}.pkl'.format(model_name, TRAINING_DATA_SIZE))
        test_predictions[model_name] = m.predict(data['test_x'])
        train_predictions[model_name] = m.predict(data['train_x'])

    final_test_preds, final_train_preds = [], []
    for i in range(len(data['test_y'])):
        preds = [test_predictions['{}-{}'.format(*pair)][i] for pair in (('THE', 'A'), ('THE', 'ZERO'), ('A', 'ZERO'))]
        counts = {'THE': 0, 'A':0, 'ZERO': 0}
        for p in preds:
            counts[p] += 1
        pred = max(counts, key=lambda x: counts[x])
        #if counts[pred] == 1:
        #    pred = 'ZERO'
        final_test_preds.append(pred)
        #if not(preds[0] == preds[1] or preds[0] == preds[2] or preds[1] == preds[2]):
        #    print(preds, data['test_y'][i])
    for i in range(len(data['train_y'])):
        preds = [train_predictions['{}-{}'.format(*pair)][i] for pair in (('THE', 'A'), ('THE', 'ZERO'), ('A', 'ZERO'))]
        counts = {'THE': 0, 'A':0, 'ZERO': 0}
        for p in preds:
            counts[p] += 1
        pred = max(counts, key=lambda x: counts[x])
        #if counts[pred] == 1:
        #    pred = 'ZERO'
        final_train_preds.append(pred)

    with open(os.path.join(CV_RESULTS_LOG_PATH, 'logreg_majvote_binarized_allfeatures_{}.txt'.format(TRAINING_DATA_SIZE)), 'a+') as f:
        f.write("Accuracy ({}): {:.4f}\n".format(TRAIN_SET_NAME, metrics.accuracy_score(data['train_y'], final_train_preds)))
        f.write("Accuracy ({}): {:.4f}\n".format(TEST_SET_NAME, metrics.accuracy_score(data['test_y'], final_test_preds)))


ART_TO_NB_MAP = {'THE': 0, 'A':1, 'ZERO': 2}
TARGET_TUPLE = ('THE', 'A', 'ZERO')
def eval_binary_models_pwc(data):

    test_predictions = np.full((data['test_x'].shape[0], 3 ** 2), 0.5)
    train_predictions = np.full((data['train_x'].shape[0], 3 ** 2), 0.5)
    data_size = np.zeros((3,3))
    for pair in (('THE', 'A'), ('THE', 'ZERO'), ('A', 'ZERO')):
        model_name = '{}-{}'.format(*pair)
        m = _load_model('logreg_binarized_{}_allfeatures_{}.pkl'.format(model_name, TRAINING_DATA_SIZE))
        p = m.predict_proba(data['test_x'])
        i, j = ART_TO_NB_MAP[m.classes_[0]], ART_TO_NB_MAP[m.classes_[1]]
        test_predictions[:, i*3 + j] = p[:, 0]
        i, j = ART_TO_NB_MAP[m.classes_[1]], ART_TO_NB_MAP[m.classes_[0]]
        test_predictions[:, i*3 + j] = p[:, 1]

        # p = m.predict_proba(data['train_x'])
        # i, j = ART_TO_NB_MAP[m.classes_[0]], ART_TO_NB_MAP[m.classes_[1]]
        # train_predictions[:, i * 3 + j] = p[:, 0]
        # i, j = ART_TO_NB_MAP[m.classes_[1]], ART_TO_NB_MAP[m.classes_[0]]
        # train_predictions[:, i * 3 + j] = p[:, 1]

        # velikost trenovacich dat pro pwc:
        train_x, train_y = _prepare_training_data(data, pair)
        data_size[i,j] = data_size[j,i] = len(train_y)

    test_predictions = [
        TARGET_TUPLE[idx] for idx in np.argmax(recount_probs_by_pwc(test_predictions, data_size), axis=1)
    ]
    # train_predictions = [
    #     TARGET_TUPLE[idx] for idx in np.argmax(recount_probs_by_pwc(train_predictions, data_size), axis=1)
    # ]

    with open(os.path.join(CV_RESULTS_LOG_PATH, 'logreg_pwc_binarized_allfeatures_{}.txt'.format(TRAINING_DATA_SIZE)), 'a+') as f:
        # f.write("Accuracy ({}): {:.4f}\n".format(TRAIN_SET_NAME, metrics.accuracy_score(data['train_y'], train_predictions)))
        f.write("Accuracy ({}): {:.4f}\n".format(TEST_SET_NAME, metrics.accuracy_score(data['test_y'], test_predictions)))


def eval_ovr_binary_models(data):
    target_tuple =  ('THE', 'A', 'ZERO')
    test_predictions = np.zeros((len(data['test_y']), 3))
    train_predictions = np.zeros((len(data['train_y']), 3))
    for i, target in enumerate(target_tuple):
        m = _load_model('logreg_ovr_binarized_{}_allfeatures_{}.pkl'.format(target, TRAINING_DATA_SIZE))
        assert m.classes_[0] == target
        test_predictions[:, i] = m.predict_proba(data['test_x'])[:, 0]
        train_predictions[:, i] = m.predict_proba(data['train_x'])[:, 0]


    test_predictions = [target_tuple[idx] for idx in np.argmax(test_predictions, axis=1)]
    train_predictions = [target_tuple[idx] for idx in np.argmax(train_predictions, axis=1)]
    with open(os.path.join(CV_RESULTS_LOG_PATH, 'logreg_ovr_binarized_allfeatures_{}.txt'.format(TRAINING_DATA_SIZE)), 'a+') as f:
        f.write("Accuracy ({}): {:.4f}\n".format(TRAIN_SET_NAME, metrics.accuracy_score(data['train_y'], train_predictions)))
        f.write("Accuracy ({}): {:.4f}\n".format(TEST_SET_NAME, metrics.accuracy_score(data['test_y'], test_predictions)))
    # tp = test_predictions
    # with open(os.path.join(CV_RESULTS_LOG_PATH, 'logreg_ovr_binarized_allfeatures_{}__.txt'.format(TRAINING_DATA_SIZE)), 'a+') as f:
    #     for c in (-0.2, -0.1, 0.1 , 0.2):
    #         tpa = [target_tuple[idx] for idx in np.argmax(np.array([tp[:,0] + c, tp[:,1], tp[:, 2]]).T, axis=1)]
    #         tpb = [target_tuple[idx] for idx in np.argmax(np.array([tp[:, 0], tp[:, 1] + c, tp[:, 2]]).T, axis=1)]
    #         tpc = [target_tuple[idx] for idx in np.argmax(np.array([tp[:, 0], tp[:, 1], tp[:, 2] + c]).T, axis=1)]
    #         #f.write("Accuracy ({}): {:.4f}\n".format(TRAIN_SET_NAME, metrics.accuracy_score(data['train_y'], train_predictions)))
    #         f.write("{}-{} Accuracy ({}): {:.4f}\n".format(c, 0, TEST_SET_NAME, metrics.accuracy_score(data['test_y'], tpa)))
    #         f.write("{}-{} Accuracy ({}): {:.4f}\n".format(c, 1, TEST_SET_NAME, metrics.accuracy_score(data['test_y'], tpb)))
    #         f.write("{}-{} Accuracy ({}): {:.4f}\n".format(c, 2, TEST_SET_NAME, metrics.accuracy_score(data['test_y'], tpc)))


def eval_ovr_outofbox_binary_model(data):
    model_name = 'logreg_ovr_outofbox_binarized_allfeatures_{}'.format(TRAINING_DATA_SIZE)
    classifier = _load_model(model_name + '.pkl')
    train_predictions = classifier.predict(data['train_x'])
    test_predictions = classifier.predict(data['test_x'])
    with open(os.path.join(CV_RESULTS_LOG_PATH, model_name + '.txt'), 'a+') as f:
        f.write(str(classifier.get_params()) + '\n')
        f.write("Accuracy ({}): {:.4f}\n".format(TRAIN_SET_NAME, metrics.accuracy_score(data['train_y'], train_predictions)))
        f.write("Accuracy ({}): {:.4f}\n".format(TEST_SET_NAME, metrics.accuracy_score(data['test_y'], test_predictions)))


if __name__ == '__main__':
    data = {}
    data['train_x'], data['train_y'] = load_postprocessed_feature_data(FEATURES, TRAIN_SET_NAME)#, sample=TRAINING_DATA_SIZE)
    assert data['train_x'].shape[0] == TRAINING_DATA_SIZE

    data['test_x'], data['test_y'] = load_postprocessed_feature_data(FEATURES, TEST_SET_NAME)
    assert data['train_x'].shape[1] == data['test_x'].shape[1]

    #multiclass_baseline_gridsearch(data)
    #binary_gridsearch(data)
    #binary_ovr_gridsearch(data)
    #binary_ovr_outofbox_gridsearch(data)

    # MULTICLASS BASELINE:
    #train_multiclass_baseline_model(data)
    eval_multiclass_baseline_model(data)

    # OVO BINARY MODELS
    #train_binary_models(data)
    eval_binary_models_majvote(data)
    eval_binary_models_pwc(data)

    # OVR BINARY MODELS
    #train_ovr_binary_models(data)
    eval_ovr_binary_models(data)

    # OVR OUTOFBOX BINARY MODELS
    #train_ovr_outofbox_binary_models(data)
    eval_ovr_outofbox_binary_model(data)
