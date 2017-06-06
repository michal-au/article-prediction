import typing
from ...lib.Tree import Tree
from ...lib.wordnet import lemmatize


def is_postmodified(bnp):
    # type: (Tree) -> bool
    return bool(bnp.parent and bnp.parent.get_label() == 'NP' and bnp.get_right_siblings())


def postmodification_length(bnp):
    # type: (Tree) -> int
    if not bnp.parent or bnp.parent.get_label() != 'NP':
        return 0
    sibs = bnp.get_right_siblings()
    if not sibs:
        return 0
    else:
        return len(sibs[0].get_words())


def postmodification_type(bnp):
    # type: (Tree) -> str
    if not bnp.parent or bnp.parent.get_label() != 'NP':
        return None
    sibs = bnp.get_right_siblings()
    if not sibs:
        return None
    else:
        return sibs[0].get_label()


def postmodification_type_specific(bnp):
    # type: (Tree) -> str
    if not bnp.parent or bnp.parent.get_label() != 'NP':
        return None
    sibs = bnp.get_right_siblings()
    if not sibs:
        return None
    else:
        postmod_token = sibs[0].get_leftmost_child()
        return postmod_token.get_word_form()
        #return lemmatize(postmod_token.get_word_form(), postmod_token.get_label())


# TODO: testme!
def postmodification_pp_of(bnp):
    # type: (Tree) -> str
    if not bnp.parent or bnp.parent.get_label() != 'NP':
        return False
    sibs = bnp.get_right_siblings()
    if not sibs:
        return False
    else:
        return sibs[0].get_label() == 'PP' and sibs[0].get_leftmost_child().get_word_form().lower() == 'of'

"""
def is_premodified(bnp):
    # type: (Tree) -> bool
    return bool(bnp.parent and bnp.parent.get_label() == 'NP' and bnp.get_left_siblings())


def premodification_length(bnp):
    # type: (Tree) -> int
    if not bnp.parent or bnp.parent.get_label() != 'NP':
        return 0
    sibs = bnp.get_left_siblings()
    if not sibs:
        return 0
    else:
        return len(sibs[-1].get_words())
"""