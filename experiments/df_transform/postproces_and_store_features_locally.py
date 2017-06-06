import numpy as np
import pandas
import os
from code.lib.utils import read_settings, save_csr_matrix
from .convert_bools import convert_bools
from .embeddings import postprocess_embeddings
from sklearn.feature_extraction import DictVectorizer
import scipy.sparse
import pickle


SETTINGS = read_settings()
CATEGORICAL_LIST_FEATURES = (
    'a_hypernyms',
    'b_pos_after_head_as_list', 'b_pos_before_head_as_list', 'b_words_after_head_as_list',
    'b_words_after_np_as_list', 'b_words_before_head_as_list', 'b_words_before_np_as_list'
)
TRAIN_DATASET_NAME, HELDOUT_DATASET_NAME, TEST_DATASET_NAME = 'train', 'heldout', 'test'
RARE_FEATURE_VALUE_CUTOFF = 5
OUT_OF_VOCABULARY_TOKEN = '-OOV-'


def postproces_dataframes(datasets):
    assert_columns_match(datasets)

    print("Cleaning fake (informative) features...")
    for dataset_type, dataset in datasets.items():
        print("- for dataset {}".format(dataset_type))
        datasets[dataset_type] = dataset.drop([f_name for f_name in dataset.columns if f_name.startswith("_")], 1)
    assert_columns_match(datasets)
    print("Done")

    print("Saving target variable...")
    for dataset_type, dataset in datasets.items():
        print("- for dataset {}".format(dataset_type))
        x, y = separate_target_variable(dataset)
        # pro dalsi praci po konci cyklu:
        datasets[dataset_type] = x

        with open(get_output_location(dataset_type, 'Y_article'), 'wb') as f:
            np.save(f, y)
    assert_columns_match(datasets)
    print("Done")

    print("Processing numerical features...")
    for dataset_type, dataset in datasets.items():
        print("- for dataset {}".format(dataset_type))
        numerical_feature_names = []
        for feature_name in dataset.columns:
            if dataset[feature_name].dtype in (np.float64, np.int64):
                print("-- for feature {}".format(feature_name))
                numerical_feature_names.append(feature_name)
                save_csr_matrix(
                    nparray_to_csr_column_vector(dataset[feature_name].values),
                    get_output_location(dataset_type, feature_name)
                )
        datasets[dataset_type] = dataset.drop(numerical_feature_names, 1)
    assert_columns_match(datasets)
    print("Done")

    print("Processing embeddings...")
    for dataset_type, dataset in datasets.items():
        x, embeddings_arrays = postprocess_embeddings(dataset)
        # pro dalsi praci po konci cyklu:
        datasets[dataset_type] = x

        for feature_name, emb_array in embeddings_arrays.iteritems():
            emb_array = emb_array.astype(np.float)
            save_csr_matrix(
                scipy.sparse.csr_matrix(emb_array),
                get_output_location(dataset_type, feature_name)
            )
    assert_columns_match(datasets)
    print("Done")

    print("Processing bools...")
    for dataset_type, dataset in datasets.items():
        x, bools_arrays = convert_bools(dataset)
        # pro dalsi praci po konci cyklu:
        datasets[dataset_type] = x

        for feature_name, b_array in bools_arrays.iteritems():
            save_csr_matrix(
                nparray_to_csr_column_vector(b_array),
                get_output_location(dataset_type, feature_name)
            )
    assert_columns_match(datasets)
    print("Done")

    print("Processing categorical list features...")
    for feature_name in CATEGORICAL_LIST_FEATURES:
        print("- for feature {}".format(feature_name))
        rare_list_feature_values_cutoff(feature_name, datasets)
        apply_dict_vectorizer_for_feature(
            feature_name,
            datasets,
            convert_column_to_list_of_dicts=lambda col: [
                {val: True for val in row_val}
                if row_val else {} for row_val in col
            ]
        )
    for dataset_type, dataset in datasets.items():
        datasets[dataset_type] = dataset.drop(list(CATEGORICAL_LIST_FEATURES), 1)
    assert_columns_match(datasets)
    print("Done")

    print("Processing categorical features...")
    for feature_name in datasets[TRAIN_DATASET_NAME].columns:
        print("- for feature {}".format(feature_name))
        print("-- handling oov words ...")
        rare_feature_values_cutoff(feature_name, datasets)
        print("-- one-hot-encoding ...")
        apply_dict_vectorizer_for_feature(
            feature_name,
            datasets,
            convert_column_to_list_of_dicts=lambda col: [{val: True} for val in col]
        )
    print("Done")


def apply_dict_vectorizer_for_feature(feature_name, datasets, convert_column_to_list_of_dicts):
    vec = DictVectorizer()

    column_as_dicts = convert_column_to_list_of_dicts(datasets[TRAIN_DATASET_NAME][feature_name])
    binarized_column = vec.fit_transform(column_as_dicts)
    pickle.dump(vec, open(os.path.join(SETTINGS.get('paths', 'modelVectorizers'), feature_name), 'wb'), protocol=2)
    save_csr_matrix(binarized_column, get_output_location(TRAIN_DATASET_NAME, feature_name))
    for dataset_name in (HELDOUT_DATASET_NAME, TEST_DATASET_NAME):
            column_as_dicts = convert_column_to_list_of_dicts(datasets[dataset_name][feature_name])
            binarized_column = vec.transform(column_as_dicts)
            save_csr_matrix(binarized_column, get_output_location(dataset_name, feature_name))


def rare_list_feature_values_cutoff(feature_name, datasets):
    # natazeni slovniku s poctem vyskytu
    column_train = datasets[TRAIN_DATASET_NAME][feature_name]
    value_counts = {}

    for val in column_train:
        if val:
            for v in val:
                value_counts[v] = value_counts[v] + 1 if value_counts.get(v) else 1
    pickle.dump(value_counts, open(os.path.join(SETTINGS.get('paths', 'modelVocabsForLists'), feature_name), 'wb'), protocol=2)
    # nahrazeni malo se vyskytujicich slov pro kazdy dataset
    for dataset_type, dataset in datasets.items():
        for idx, val in enumerate(dataset[feature_name]):
            if val:
                new_val = []
                for v in val:
                    if value_counts.get(v, 0) <= RARE_FEATURE_VALUE_CUTOFF:
                        new_val.append(OUT_OF_VOCABULARY_TOKEN)
                    else:
                        new_val.append(v)
                dataset.set_value(idx, feature_name, new_val)


def rare_feature_values_cutoff(feature_name, datasets):
    column_train = datasets[TRAIN_DATASET_NAME][feature_name]
    column_train.fillna('missing_value', inplace=True)
    freq_dist = column_train.value_counts()

    value_counts = {}
    for val in column_train:
        if val:
            value_counts[val] = value_counts[val] + 1 if value_counts.get(val) else 1
    pickle.dump(value_counts, open(os.path.join(SETTINGS.get('paths', 'modelVocabs'), feature_name), 'wb'), protocol=2)

    to_replace = freq_dist[freq_dist <= RARE_FEATURE_VALUE_CUTOFF].index
    if len(to_replace):
        column_train.replace(to_replace, OUT_OF_VOCABULARY_TOKEN, inplace=True)
    for dataset_name in (HELDOUT_DATASET_NAME, TEST_DATASET_NAME):
        column = datasets[dataset_name][feature_name]
        column.fillna('missing_value', inplace=True)
        to_replace = column[(freq_dist[column].fillna(0) <= RARE_FEATURE_VALUE_CUTOFF).values].values
        if len(to_replace):
            column.replace(to_replace, OUT_OF_VOCABULARY_TOKEN, inplace=True)


def separate_target_variable(df):
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


def get_output_location(dataset_type, feature_name):
    """
    pro zadany dataset (train/heldout/test) a featuru vrati cestu k ulozeni
    """
    return os.path.join(
        SETTINGS.get('paths', 'dataFeaturesPennPostprocessedTMP'),
        dataset_type,
        feature_name,
    )


def nparray_to_csr_column_vector(nparray):
    nparray.shape = (len(nparray), 1)
    return scipy.sparse.csr_matrix(nparray)


def assert_columns_match(datasets):
    assert (
        datasets[TRAIN_DATASET_NAME].dtypes.equals(datasets[HELDOUT_DATASET_NAME].dtypes)
        and
        datasets[HELDOUT_DATASET_NAME].dtypes.equals(datasets[TEST_DATASET_NAME].dtypes)
    )
    print("columns match assertion: OK")


if __name__ == '__main__':
    settings = read_settings()

    print("Reading datasets...")
    datasets = {
        TRAIN_DATASET_NAME: pandas.read_pickle(settings.get('paths', 'dataFeaturesPennTrain')),
        HELDOUT_DATASET_NAME: pandas.read_pickle(SETTINGS.get('paths', 'dataFeaturesPennHeldout')),
        TEST_DATASET_NAME: pandas.read_pickle(settings.get('paths', 'dataFeaturesPennTest'))
    }
    print("Done")

    postproces_dataframes(datasets)