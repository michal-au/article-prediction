import os
import pandas
import codecs
from collections import deque
from gensim.models import word2vec
import cPickle

from ..lib.Tree import Tree
from ..lib.wordnet import lemmatize
from ..lib import utils
from ..lib.sentence import SentenceStateTracker
from ..lib.language_model.utils.sentence_preprocess import preprocess_sent_for_zero_lc_nb, preprocess_sent_for_lc_nb
from ..lib.language_model import kenlm
from .np_selector import collins_selector, penn_selector
from .lee_methods.feature_methods import *
from .lee_methods.feature_methods_modified import *
from .additional_methods.head_proper import head_pos_simple, head_proper
from .additional_methods.others import (
    non_article_dets_zero_marker,
    relative_position_within_sent,
    extended_referent,
    extended_referent_with_propers,
    smoothed_parent,
)
from .additional_methods.predicate import (
    main_verb_form,
    subject_verb_form,
    subject_position,
    object_verb_form,
    pp_object_form,
)
from .additional_methods.np_modification import (
    is_postmodified,
    postmodification_type,
    postmodification_type_specific,
    postmodification_length,
    postmodification_pp_of,
)
from countability.countability import countability, referent_with_countability
from embeddings.embeddings import head_form_embeddings, embed_feature_selection  # embed_poly, object_form_embeddings, avg_embeddings

from language_model.kenlm import score_by_kenlm


__all__ = ('extract_features_from_file')


SETTINGS = utils.read_settings()


print "reading decision lists"
DECISION_LISTS_BNC = cPickle.load(open(os.path.join(SETTINGS.get('paths', 'dataCountability'), 'decision_lists-bnc.pkl'), 'r'))

print "reading embeddings"
EMBEDDINGS = word2vec.Word2Vec.load_word2vec_format(SETTINGS.get('paths', 'dataEmbeddingsGoogle'), binary=True)

#INDICIES_F = cPickle.load(open(os.path.join(SETTINGS.get('paths', 'dataEmbeddingsIndicies'), 'indicies_f_crossvalid.pkl'), 'rb'))
#INDICIES_F = sorted(INDICIES_F, key=lambda x: INDICIES_F[x], reverse=True)[:300]
#INDICIES_MI = cPickle.load(open(os.path.join(SETTINGS.get('paths', 'dataEmbeddingsIndicies'), 'indicies_mi.pkl'), 'rb'))


# LANGUAGE MODELS:
print "reading language models"
#model_kenlm_ptb_10_lc_nbs_zeros = kenlm.Model(os.path.join(SETTINGS.get('paths', 'modelLM'), 'kenlm-ptb-10-lc-nbs-zeros'))
#model_kenlm_ggl_10_lc_nbs_zeros = kenlm.Model(os.path.join(SETTINGS.get('paths', 'modelLM'), 'kenlm-ggl-10-lc-nbs-zeros'))
#model_kenlm_ptb_5_lc_nbs = kenlm.Model(os.path.join(SETTINGS.get('paths', 'modelLM'), 'kenlm-ptb-5-lc-nbs'))
model_kenlm_ggl_5_nbs = kenlm.Model(os.path.join(SETTINGS.get('paths', 'modelLM'), 'kenlm-ggl-5-nbs-cls3'))


def _add_to_history(history, t):
    # type (history: deque, t: Tree) -> deque
    history.popleft()
    word_forms = [lemmatize(wtp[0], tag=wtp[1]) for wtp in t.get_word_tag_pairs()]  # todo: .lowercase???
    history.append(word_forms)
    return history


def populate_feature_dict(bnp, history, context_words, article_length, node_nb, lms_sentence, is_test=False):
    h_form = head_form(bnp)
    _words_before_head = words_before_head(bnp)
    _words_after_head = words_after_head(bnp)
    _words_before_np = words_before_np(bnp)
    _words_after_np = words_after_np(bnp)
    feature_dict = {
        'Y_article': str(article_class(bnp)),
        'a_head_form': h_form,
        'a_head_number': head_number(bnp),
        'a_head_pos': head_pos(bnp),
        'a_parent': parent(bnp),
        'a_non_article_det': non_article_det(bnp),
        'a_words_before_head': _words_before_head,
        'a_pos_before_head': pos_before_head(bnp),
        'a_words_after_head': _words_after_head,
        'a_pos_after_head': pos_after_head(bnp),
        'a_words_before_np': _words_before_np,
        'a_words_after_np': _words_after_np,
        'a_hypernyms': hypernyms(h_form),
        'a_referent': referent(h_form, history),
    }
    feature_dict.update({
        'b_words_before_head_as_list': words_before_head_as_list(bnp),
        'b_pos_before_head_as_list': pos_before_head_as_list(bnp),
        'b_words_after_head_as_list': words_after_head_as_list(bnp),
        'b_pos_after_head_as_list': pos_after_head_as_list(bnp),
        'b_words_before_np_as_list': words_before_np_as_list(bnp),
        'b_words_after_np_as_list': words_after_np_as_list(bnp),
    })
    feature_dict.update({
        'b_head_proper': head_proper(bnp),
        'b_head_pos_simple': head_pos_simple(bnp),
        'b_is_postmodified': is_postmodified(bnp),
        'b_postmodification_type': postmodification_type(bnp),
        'b_postmodification_type_specific': postmodification_type_specific(bnp),
        'b_postmodification_length': postmodification_length(bnp),
        'b_postmodification_pp_of': postmodification_pp_of(bnp),
        'b_non_article_det_extended': non_article_dets_zero_marker(bnp),
        'b_relative_position': relative_position_within_sent(bnp),
        #'b_predicate_form': main_verb_form(bnp),
        'b_referent': extended_referent(h_form, bnp, history),
        'b_referent_with_propers': extended_referent_with_propers(h_form, bnp, history),
        'b_subject_verb_form': subject_verb_form(bnp),
        'b_subject_position': subject_position(bnp),
        'b_parent': parent(bnp) if is_test else smoothed_parent(bnp),
        'b_object_form': object_verb_form(bnp),
        'b_pp_object_form': pp_object_form(bnp),
        'b_position_within_article': node_nb/float(article_length),
        #'c_countability': countability(bnp, DECISION_LISTS, context_words),
        'c_countability_bnc': countability(bnp, DECISION_LISTS_BNC, context_words),
        #'c_referent_with_countability': referent_with_countability(bnp, DECISION_LISTS_BNC, context_words, extended_referent(h_form, bnp, history)),
        #'c_referent_with_countability_with_propers': referent_with_countability(bnp, DECISION_LISTS_BNC, context_words, extended_referent_with_propers(h_form, bnp, history)),
        'd_head_form_embeddings': head_form_embeddings(EMBEDDINGS, h_form),
        #'d_head_form_embeddings_selection_f': embed_feature_selection(EMBEDDINGS, h_form, INDICIES_F),
        #'d_head_form_embeddings_selection_mi': embed_feature_selection(EMBEDDINGS, h_form, INDICIES_MI),
        #'d_object_form_embeddings': object_form_embeddings(EMBEDDINGS, bnp),
        #'d_words_before_head_embeddings': avg_embeddings(EMBEDDINGS, _words_before_head),
        #'d_words_after_head_embeddings': avg_embeddings(EMBEDDINGS, _words_after_head),
        #'d_words_before_np_embeddings': avg_embeddings(EMBEDDINGS, _words_before_np),
        #'d_words_after_np_embeddings': avg_embeddings(EMBEDDINGS, _words_after_np),
        #'e_kenlm_ptb_10_lc_nbs_zeros': score_by_kenlm(model_kenlm_ptb_10_lc_nbs_zeros, bnp, lms_sentence['ptb-10-lc-nbs-zeros'], prob=True),
        #'e_kenlm_ggl_10_lc_nbs_zeros': score_by_kenlm(model_kenlm_ggl_10_lc_nbs_zeros, bnp, lms_sentence['ggl-10-lc-nbs-zeros'], prob=True),
        #'e_kenlm_ptb_5_lc_nbs': score_by_kenlm(model_kenlm_ptb_5_lc_nbs, bnp, lms_sentence['ptb-5-lc-nbs'], prob=False),
        'e_kenlm_ggl_5_lc_nbs': score_by_kenlm(model_kenlm_ggl_5_nbs, bnp, lms_sentence['ggl-5-lc-nbs'], prob=False),
    })
    #print "pred vals", feature_dict['e-kenlm-ptb-10-lc-nbs-zeros']
    #print "pred vals", feature_dict['e-kenlm-ggl-10-lc-nbs-zeros']
    #print "pred vals", feature_dict['e-kenlm-ptb-5-lc-nbs']
    #print "pred vals", feature_dict['e-kenlm-ggl-5-lc-nbs']
    return feature_dict


def extract_features_from_file(f_path, np_selector=collins_selector, is_test=False):
    # type (f_path: string) -> List(List())
    feature_collection = []
    print f_path
    with codecs.open(f_path, 'r', 'utf-8') as f:
        history = deque([None] * 5)
        lines = f.readlines()
        for l_num, l in enumerate(lines, start=1):
            t = Tree.from_string(l)
            context_words = t.get_word_tag_pairs()
            tokens_lc_nbs_zeros = preprocess_sent_for_zero_lc_nb(t.get_words())
            tokens_lc_nbs = preprocess_sent_for_lc_nb(t.get_words())
            lms_sentence = {
                'ptb-10-lc-nbs-zeros': SentenceStateTracker(tokens_lc_nbs_zeros, zero_filled=True),
                'ggl-10-lc-nbs-zeros': SentenceStateTracker(tokens_lc_nbs_zeros, zero_filled=True),
                'ptb-5-lc-nbs': SentenceStateTracker(tokens_lc_nbs, zero_filled=False),
                'ggl-5-lc-nbs': SentenceStateTracker(tokens_lc_nbs, zero_filled=False),
            }  # continuously updated sentence for language model article prediction
            for n_num, node in enumerate(t, start=1):
                if np_selector(node):
                    example_features = populate_feature_dict(
                        node, history, context_words, article_length=len(lines), node_nb=l_num - 1, is_test=is_test,
                        lms_sentence=lms_sentence,
                    )
                    example_features['_coordinates'] = f_path + "|" + str(l_num) + "|" + str(n_num)
                    example_features['_sent'] = node.print_highlighted()
                    feature_collection.append(example_features)
            history = _add_to_history(history, t)

    return feature_collection


def _save_data_frame(data_frame, f_name):
    out_path = SETTINGS.get('paths', 'dataFeatures')
    pickle_path = os.path.join(out_path, f_name + '.pkl')
    utils.create_dir_for_file(pickle_path)
    data_frame.to_pickle(pickle_path)


if __name__ == '__main__':
    # EXTRACTING TRAINING FEATURES:
    path = SETTINGS.get('paths', 'dataParsed')
    feature_collection = []  # the resulting feature table
    for r, ds, fs in os.walk(path):
        if r in [os.path.join(path, dir_nb) for dir_nb in ('22', '23', '24')]:
            continue
        print r
        ds.sort()
        fs.sort()
        for f in fs:
            f_path = os.path.join(r, f)
            feature_collection.extend(extract_features_from_file(f_path))

    df = pandas.DataFrame(feature_collection)
    _save_data_frame(df, 'penn/train')

    # EXTRACTING HELDOUT DATA FEATURES:
    path = SETTINGS.get('paths', 'dataParsed')
    feature_collection = []  # the resulting feature table
    for r, ds, fs in os.walk(os.path.join(path, '22')):
        fs.sort()
        for f in fs:
            f_path = os.path.join(r, f)
            feature_collection.extend(extract_features_from_file(f_path))

    df = pandas.DataFrame(feature_collection)
    _save_data_frame(df, 'penn/heldout')

    # EXTRACTING TEST FEATURES:
    path = SETTINGS.get('paths', 'dataParsedOrig')
    feature_collection = []  # the resulting feature table
    for r, ds, fs in os.walk(os.path.join(path, '23')):
        fs.sort()
        for f in fs:
            f_path = os.path.join(r, f)
            feature_collection.extend(extract_features_from_file(
                f_path,
                np_selector=penn_selector,
                is_test=True,
            ))

    df = pandas.DataFrame(feature_collection)
    _save_data_frame(df, 'penn/test')
