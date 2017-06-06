"""
Vytahne featury pro dany seznam stromu. Chtelo by zintegrovat se stejnym procesem pro penn experimenty
"""
import pandas as pd
import numpy as np
import scipy
from collections import deque
from code.extract_features.extract import populate_feature_dict, _add_to_history
from code.lib import utils
from code.lib.sentence import SentenceStateTracker
import os
from code.experiments.features import get_features_for_set_names
from code.lib.language_model.utils.sentence_preprocess import process_nbs
from code.experiments.df_transform.postproces_and_store_features_locally import nparray_to_csr_column_vector, CATEGORICAL_LIST_FEATURES, RARE_FEATURE_VALUE_CUTOFF, OUT_OF_VOCABULARY_TOKEN
from code.experiments.df_transform.embeddings import postprocess_embeddings
from code.experiments.df_transform.convert_bools import convert_bools
import pickle


SETTINGS = utils.read_settings()
ALL_FEATURE_SETS = get_features_for_set_names(('orig', 'extended', 'countability', 'embeddings', 'lm'))
VECTORIZERS_PATH = os.path.join(SETTINGS.get('paths', 'modelLM'), 'vectorizers')


def np_selector(n):
    """Duplikat k code.extract_features.extract_bnc.np_selector
    naimportovanim originalu bychom ale zacali nacitat vsechny embeddingy, decision listy atd..."""
    if n.get_label() != "NP":
        return False
    if not n.children:
        return False
    if "NP" not in [ch.get_label() for ch in n.children]:
        return True
    np_children = [ch for ch in n.children if ch.get_label() == "NP"]
    return all(['POS' in [chch.get_label() for chch in np_child.children] for np_child in np_children])


# TMP - smazat
def _save_data_frame(data_frame, f_name):
    pickle_path = os.path.join('code', 'corrector', f_name + '.pkl')
    data_frame.to_pickle(pickle_path)


def extract_features(trees, bnc=False):
    feature_collection = []
    history = deque([None] * 5)

    for t_num, t in enumerate(trees, start=1):
        context_words = t.get_word_tag_pairs()

        sent = [process_nbs(token) for token in t.get_words()]
        lang_model_sent = {
            'ggl-5-lc-nbs': SentenceStateTracker(sent, zero_filled=False),
        }  # continuously updated sentence for language model article prediction

        for n_num, node in enumerate(t, start=1):
            if np_selector(node):
                example_features = populate_feature_dict(node, history, context_words, article_length=len(trees), node_nb=t_num - 1, lms_sentence=lang_model_sent, is_test=False)
                example_features['_sent'] = node.print_highlighted()
                feature_collection.append(example_features)

            history = _add_to_history(history, t)

    df = pd.DataFrame(feature_collection)
    _save_data_frame(df, 'tmp_bnc_dataframe')
    df = None
    test_x = postprocess_features(df, bnc=bnc)
    return test_x


def postprocess_features(df, bnc=False):
    # TODO: smazat
    df = pd.read_pickle(os.path.join('code', 'corrector', 'tmp_bnc_dataframe.pkl'))
    ready_features = {}

    features_to_drop = [f for f in df.columns if f not in ALL_FEATURE_SETS]
    df = df.drop(features_to_drop, 1)

    numerical_feature_names = []
    for feature_name in df.columns:
        if df[feature_name].dtype in (np.float64, np.int64):
            numerical_feature_names.append(feature_name)
            ready_features[feature_name] = nparray_to_csr_column_vector(df[feature_name].values),
    df = df.drop(numerical_feature_names, 1)

    df, embeddings_arrays = postprocess_embeddings(df)
    for feature_name, emb_array in embeddings_arrays.iteritems():
        emb_array = emb_array.astype(np.float)
        ready_features[feature_name] = scipy.sparse.csr_matrix(emb_array)

    df, bools_arrays = convert_bools(df)
    for feature_name, b_array in bools_arrays.iteritems():
        ready_features[feature_name] = nparray_to_csr_column_vector(b_array)

    vocab, vecs = load_vectorizers(df.columns, bnc=bnc)
    for feature_name in df.columns:
        value_counts = vocab.get(feature_name)
        if feature_name in CATEGORICAL_LIST_FEATURES:
            for idx, val in enumerate(df[feature_name]):
                if val:
                    new_val = []
                    for v in val:
                        if value_counts.get(v, 0) <= RARE_FEATURE_VALUE_CUTOFF:
                            new_val.append(OUT_OF_VOCABULARY_TOKEN)
                        else:
                            new_val.append(v)
                    df.set_value(idx, feature_name, new_val)
            column_as_dicts = [{val: True for val in row_val} if row_val else {} for row_val in df[feature_name]]
            ready_features[feature_name] = vecs.get(feature_name).transform(column_as_dicts)
        else:
            for idx, val in enumerate(df[feature_name]):
                if val:
                    if value_counts.get(val, 0) <= RARE_FEATURE_VALUE_CUTOFF:
                        new_val = OUT_OF_VOCABULARY_TOKEN
                    else:
                        new_val = val
                    df.set_value(idx, feature_name, new_val)
            column_as_dicts = [{val: True} for val in df[feature_name]]
            ready_features[feature_name] = vecs.get(feature_name).transform(column_as_dicts)
    test_x = scipy.sparse.hstack([ready_features[f] for f in ALL_FEATURE_SETS])
    print test_x.shape
    return test_x


def load_vectorizers(features, bnc=False):
    vocabs = {
        f_name: pickle.load(
            open(os.path.join(SETTINGS.get('paths', 'modelBncVocabs' if bnc else 'modelVocabsForLists'), f_name), 'rb'),
        ) for f_name in features if f_name in CATEGORICAL_LIST_FEATURES
    }
    vocabs.update({
        f_name: pickle.load(
            open(os.path.join(SETTINGS.get('paths', 'modelBncVocabs' if bnc else 'modelVocabs'), f_name), 'rb'),
        ) for f_name in features if f_name not in CATEGORICAL_LIST_FEATURES
    })
    vecs = {
        f_name: pickle.load(
            open(os.path.join(SETTINGS.get('paths', 'modelBncVectorizers' if bnc else 'modelVectorizers'), f_name), 'rb'),
        ) for f_name in features
    }
    return vocabs, vecs
