import typing
from typing import Tuple

from ...lib.wordnet import lemmatize, hypernyms  # type: ignore
from ...lib.wordnet import hypernyms as hyper  # type: ignore
from ...lib.Tree import Tree
from ...lib.Articles import Article


def word_suitable(wtp):
    return not wtp[0].lower() in ('the', 'a', 'an')


def words_before_head_as_list(bnp):
    # type: (Tree) -> List[str]
    """
    lemmas of fords within the np before the head, (excludes determiners)
    """
    word_tag_pairs = []
    h = bnp.get_head_collins()
    for n in h.get_left_siblings():
        word_tag_pairs.extend(n.get_word_tag_pairs())
    return [lemmatize(wtp[0], wtp[1]) for wtp in word_tag_pairs if word_suitable(wtp)]
    #return [lemmatize(wtp[0], wtp[1]) for wtp in word_tag_pairs if wtp[1] != 'DT']
    #return [lemmatize(wtp[0], wtp[1]) for wtp in word_tag_pairs if not wtp[0].lower() in ('the', 'a', 'an')][-2:]


def words_after_head_as_list(bnp):
    # type: (Tree) -> List[str]
    """
    lemmas of fords within the np after the head, (excludes determiners)
    """
    word_tag_pairs = []
    h = bnp.get_head_collins()
    for n in h.get_right_siblings():
        word_tag_pairs.extend(n.get_word_tag_pairs())
    return [lemmatize(wtp[0], wtp[1]) for wtp in word_tag_pairs if word_suitable(wtp)]
    #return [lemmatize(wtp[0], wtp[1]) for wtp in word_tag_pairs if wtp[1] != 'DT']
    #return [lemmatize(wtp[0], wtp[1]) for wtp in word_tag_pairs if not wtp[0].lower() in ('the', 'a', 'an')][:2]


def pos_before_head_as_list(bnp):
    # type: (Tree) -> List[str]
    """
    pos of fords within the np before the head, (excludes determiners)
    """
    word_tag_pairs = []
    h = bnp.get_head_collins()
    for n in h.get_left_siblings():
        word_tag_pairs.extend(n.get_word_tag_pairs())
    return [wtp[1] for wtp in word_tag_pairs if word_suitable(wtp)]
    #return [wtp[1] for wtp in word_tag_pairs if wtp[1] != 'DT']
    #return [wtp[1] for wtp in word_tag_pairs if not wtp[0].lower() in ('the', 'a', 'an')][-2:]


def pos_after_head_as_list(bnp):
    # type: (Tree) -> List[str]
    """
    pos of fords within the np after the head, (excludes determiners)
    """
    word_tag_pairs = []
    h = bnp.get_head_collins()
    for n in h.get_right_siblings():
        word_tag_pairs.extend(n.get_word_tag_pairs())
    return [wtp[1] for wtp in word_tag_pairs if word_suitable(wtp)]
    #return [wtp[1] for wtp in word_tag_pairs if wtp[1] != 'DT']
    #return [wtp[1] for wtp in word_tag_pairs if not wtp[0].lower() in ('the', 'a', 'an')][:2]


def words_before_np_as_list(bnp):
    # type: (Tree) -> List[str]
    """
    lemmas of two words preceding the np
    """
    leftmost_child = bnp.get_leftmost_child()
    idx = leftmost_child.order_nb
    root = bnp.get_root()
    leaves = root.get_word_tag_pairs()
    return [
        lemmatize(wtp[0], wtp[1]) for wtp in leaves[max(0, idx - 2): idx]
        if wtp[0].lower() not in ('the', 'a', 'an')
    ]


def words_after_np_as_list(bnp):
    # type: (Tree) -> List[str]
    """
    lemmas of two words following the np
    """
    rightmost_child = bnp.get_rightmost_child()
    idx = rightmost_child.order_nb
    root = bnp.get_root()
    leaves = root.get_word_tag_pairs()
    return [lemmatize(wtp[0], wtp[1]) for wtp in leaves[idx + 1: idx + 3] if wtp[0].lower() not in ('the', 'a', 'an')]
