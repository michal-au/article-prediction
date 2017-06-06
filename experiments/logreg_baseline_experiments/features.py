import os
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from code.lib.utils import load_postprocessed_feature_data, logreg_gridsearch
from code.lib.utils import read_settings
from code.experiments.features import get_features_for_set_names
import argparse


TRAIN_SET_NAME = 'train'
HELDOUT_SET_NAME = 'heldout'
TEST_SET_NAME = 'test'
SETTINGS = read_settings()
CV_RESULTS_LOG_PATH = os.path.join(SETTINGS.get('paths', 'logModelResults'), 'penn', 'logreg_features')

DEFAULT_MODEL_PARAMS = model_params = {
    'multi_class': 'multinomial',
    'random_state': 42,
    'penalty': 'l2',
    'solver': 'lbfgs',
}

FEATURE_SETS_COMBINATIONS = (
    ('orig', 'extended'), ('orig', 'countability'), ('orig', 'embeddings'), ('orig', 'lm'),
    ('orig', 'extended', 'countability'), ('orig', 'extended', 'countability', 'embeddings'),
    ('orig', 'extended', 'countability', 'embeddings', 'lm'),
)

FEATURE_SETS_TO_REGULARIZATION_PARAMETER_MAP = { # Vysledek kroku 'GRID SEARCH'
    # Pro cela trenovaci data (263088):
    'orig-extended': -1,
    'orig-countability': -1,
    'orig-embeddings': -1,
    'orig-lm': -1,
    'orig-extended-countability': -1,
    'orig-extended-countability-embeddings': -1,
    'orig-extended-countability-embeddings-lm': -1
}


def gridsearch(data, model_name):
    print("Gridsearch for {}".format(model_name))
    tuning_params = {
        'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    }
    logreg_gridsearch(
        data['train_x'],
        data['train_y'],
        os.path.join(CV_RESULTS_LOG_PATH, model_name + '.csv'),
        DEFAULT_MODEL_PARAMS,
        tuning_params,
    )


def train_and_eval(data, model_name, regularization_parameter):
    model_params = dict(DEFAULT_MODEL_PARAMS, C=regularization_parameter)
    model = LogisticRegression(**model_params)
    model.fit(data['train_x'], data['train_y'])

    train_predictions = model.predict(data['train_x'])
    heldout_predictions = model.predict(data['heldout_x'])
    test_predictions = model.predict(data['test_x'])
    with open(os.path.join(CV_RESULTS_LOG_PATH, model_name + '.txt'), 'a+') as f:
        f.write(str(model.get_params()) + '\n')
        f.write("Accuracy ({}): {:.4f}\n".format(TRAIN_SET_NAME, metrics.accuracy_score(data['train_y'], train_predictions)))
        f.write("Accuracy ({}): {:.4f}\n".format(HELDOUT_SET_NAME, metrics.accuracy_score(data['heldout_y'], heldout_predictions)))
        f.write("Accuracy ({}): {:.4f}\n".format(TEST_SET_NAME, metrics.accuracy_score(data['test_y'], test_predictions)))


def load_data(data_size, features):
    data = {}
    if data_size != 263088:
        data['train_x'], data['train_y'] = load_postprocessed_feature_data(features, TRAIN_SET_NAME, sample=data_size)
    else:
        data['train_x'], data['train_y'] = load_postprocessed_feature_data(features, TRAIN_SET_NAME)
    assert data['train_x'].shape[0] == data_size

    data['heldout_x'], data['heldout_y'] = load_postprocessed_feature_data(features, HELDOUT_SET_NAME)
    data['test_x'], data['test_y'] = load_postprocessed_feature_data(features, TEST_SET_NAME)
    assert data['train_x'].shape[1] == data['test_x'].shape[1]
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess text before it is used for language model')
    parser.add_argument('--datasize', required=True, help='')
    parser.add_argument('--train-and-eval', action='store_true', help='')
    parser.add_argument('--gridsearch', action='store_true', help='')
    args = parser.parse_args()

    data_size = int(args.datasize)

    if args.gridsearch:
        for combination in FEATURE_SETS_COMBINATIONS:
            features = get_features_for_set_names(combination)
            data = load_data(data_size, features)
            model_name = '{}_size_{}'.format('-'.join(combination), data_size)
            gridsearch(data, model_name)
    elif args.train_and_eval:
        for combination in FEATURE_SETS_COMBINATIONS:
            features = get_features_for_set_names(combination)
            data = load_data(data_size, features)
            model_name = '{}_size_{}'.format('-'.join(combination), data_size)
            train_and_eval(data, model_name, FEATURE_SETS_TO_REGULARIZATION_PARAMETER_MAP['-'.join(combination)])
    else:
        raise Exception("What should I do?")
