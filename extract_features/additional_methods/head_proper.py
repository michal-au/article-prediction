import typing
from ...lib.Tree import Tree


def head_pos_simple(bnp):
    # type: (Tree) -> str
    h = bnp.get_head_collins()
    label = h.get_label()
    return 'NN' if label in ('NNP', 'NNPS', 'NNS') else label


def head_proper(bnp):
    # type: (Tree) -> bool
    h = bnp.get_head_collins()
    return h.get_label() in ['NNP', 'NNPS']
