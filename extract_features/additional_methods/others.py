from __future__ import division

import typing
#from ...lib.Tree import Tree
from ...lib.wordnet import lemmatize


def non_article_dets_zero_marker(bnp):
    # type: (Tree) -> bool
    """
    rozsireni fce z lee_methods o privlastnovaci zajmena
    """
    for ch in bnp.children:
        if ch.is_leaf() and (
            (ch.get_label() == 'DT' and ch.get_word_form().lower() not in ('the', 'a', 'an')) or
            ch.get_label == 'PRP$'
        ):
            return True
    return False


def relative_position_within_sent(bnp):
    root = bnp.get_root()
    return bnp.get_leftmost_child().order_nb / (len(root.get_words()) - len(bnp.get_words()) + 1)


def extended_referent(head_form, bnp, history):
    """
    Does the head_form appear in any of the 5 preceding sentences or precedes this occurrence in the given sentence?
    """
    h = bnp.get_head_collins()
    if h.get_label() in ['PRP', 'CD', '.', 'QP', 'DT', '$', ',', "''", '``', 'NNP', 'NNPS']:
        return None

    this_sent_word_forms = [lemmatize(wtp[0], tag=wtp[1]) for wtp in bnp.get_root().get_word_tag_pairs()]
    return head_form in [word for sentence in history if sentence for word in sentence] or head_form in this_sent_word_forms[:h.order_nb]


def extended_referent_with_propers(head_form, bnp, history):
    """
    Does the head_form appear in any of the 5 preceding sentences or precedes this occurrence in the given sentence?
    """
    h = bnp.get_head_collins()
    if h.get_label() in ['PRP', 'CD', '.', 'QP', 'DT', '$', ',', "''", '``']:
        return None

    this_sent_word_forms = [lemmatize(wtp[0], tag=wtp[1]) for wtp in bnp.get_root().get_word_tag_pairs()]
    return head_form in [word for sentence in history if sentence for word in sentence] or head_form in this_sent_word_forms[:h.order_nb]


def smoothed_parent(bnp):  #TODO: testit
    if not bnp.parent:
        return None
    if bnp.parent.parent and bnp.parent.get_label() == 'NP':
        return bnp.parent.parent.get_label()
    return bnp.parent.get_label()
