import typing
from typing import Tuple

from ...lib.wordnet import lemmatize, hypernyms  # type: ignore
from ...lib.wordnet import hypernyms as hyper  # type: ignore
from ...lib.Tree import Tree
from ...lib.Articles import Article


def article_class(bnp):
    # type: (Tree) -> Article
    """
    the target class - the | a/an | 0.
    """
    for ch in bnp.children:
        if ch.is_leaf() and ch.get_label() == 'DT':
            if ch.get_word_form().lower() == 'the':
                return Article.THE
            elif ch.get_word_form().lower() in ('a', 'an'):
                return Article.A
    return Article.ZERO


def head_form(bnp):
    # type: (Tree) -> str
    """
    the root form of the head of the NP. (numbers are converted to '<number>' token)
    """
    if bnp.is_leaf():
        return None
    h = bnp.get_head_collins()
    if not h:
        print bnp
        return None
    return lemmatize(h.get_word_form(), tag=h.get_label())  # TODO: decide? .lower()
    # return h.get_word_form()


def head_number(bnp):
    # type: (Tree) -> str
    """
    the grammatical number of the NP (sg/pl)
    """
    h = bnp.get_head_collins()
    if h.get_label() in ('NN', 'NNP'):
        return 'sg'
    elif h.get_label() in ('NNS', 'NNPS'):
        return 'pl'


def head_pos(bnp):
    # type: (Tree) -> str
    """
    the POS tag of the head of the NP
    """
    if bnp.is_leaf():
        return None
    h = bnp.get_head_collins()
    label = h.get_label()
    if label == 'NNS':
        label = 'NN'
    elif label == 'NNPS':
        label = 'NNP'

    return label


def parent(bnp):
    # type: (Tree) -> str
    """
    the POS tag of the parent of the NP
    """
    if bnp.is_root():
        return None
    return bnp.parent.get_label()


def non_article_det(bnp):
    # type: (Tree) -> str
    """
    the determiner other than an article (e.g. that)
    """
    for ch in bnp.children:
        if ch.is_leaf() and ch.get_label() == 'DT' and ch.get_word_form().lower() not in ('the', 'a', 'an'):
            return ch.get_word_form().lower()


def words_before_head(bnp):
    # type: (Tree) -> List[str]
    """
    lemmas of fords within the np before the head, (excludes determiners)
    """
    word_tag_pairs = []
    h = bnp.get_head_collins()
    for n in h.get_left_siblings():
        word_tag_pairs.extend(n.get_word_tag_pairs())
    return ' '.join([lemmatize(wtp[0], wtp[1]) for wtp in word_tag_pairs if not wtp[0].lower() in ('the', 'a', 'an')])


def words_after_head(bnp):
    # type: (Tree) -> List[str]
    """
    lemmas of fords within the np after the head, (excludes determiners)
    """
    word_tag_pairs = []
    h = bnp.get_head_collins()
    for n in h.get_right_siblings():
        word_tag_pairs.extend(n.get_word_tag_pairs())
    return ' '.join([lemmatize(wtp[0], wtp[1]) for wtp in word_tag_pairs if not wtp[0].lower() in ('the', 'a', 'an')])


def pos_before_head(bnp):
    # type: (Tree) -> List[str]
    """
    pos of fords within the np before the head, (excludes determiners)
    """
    word_tag_pairs = []
    h = bnp.get_head_collins()
    for n in h.get_left_siblings():
        word_tag_pairs.extend(n.get_word_tag_pairs())
    return ' '.join([wtp[1] for wtp in word_tag_pairs if not wtp[0].lower() in ('the', 'a', 'an')])


def pos_after_head(bnp):
    # type: (Tree) -> List[str]
    """
    pos of fords within the np after the head, (excludes determiners)
    """
    word_tag_pairs = []
    h = bnp.get_head_collins()
    for n in h.get_right_siblings():
        word_tag_pairs.extend(n.get_word_tag_pairs())
    return ' '.join([wtp[1] for wtp in word_tag_pairs if not wtp[0].lower() in ('the', 'a', 'an')])


def words_before_np(bnp):
    # type: (Tree) -> List[str]
    """
    lemmas of two words preceding the np
    """
    leftmost_child = bnp.get_leftmost_child()
    idx = leftmost_child.order_nb
    root = bnp.get_root()
    leaves = root.get_word_tag_pairs()
    return ' '.join([lemmatize(wtp[0], wtp[1]) for wtp in leaves[max(0, idx - 2): idx] if wtp[0].lower() not in ('the', 'a', 'an')])


def words_after_np(bnp):
    # type: (Tree) -> List[str]
    """
    lemmas of two words following the np
    """
    rightmost_child = bnp.get_rightmost_child()
    idx = rightmost_child.order_nb
    root = bnp.get_root()
    leaves = root.get_word_tag_pairs()
    return ' '.join([lemmatize(wtp[0], wtp[1]) for wtp in leaves[idx + 1: idx + 3] if wtp[0].lower() not in ('the', 'a', 'an')])


def hypernyms(bnp):
    # type: (Tree) -> List[str]
    """
    returns wordnet synsets for the first meaning given for the head noun
    """
    return hyper(bnp.get_head_collins().get_word_form())


def referent(head_form, history):
    # type: (str, Tuple[List[str], List[str], List[str], List[str], List[str]]) -> bool
    """
    Does the head_form appear in any of the 5 preceding sentences?
    """
    return head_form in [word for sentence in history if sentence for word in sentence]
