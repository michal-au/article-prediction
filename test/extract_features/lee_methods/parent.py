from ....extract_features.lee_methods.feature_methods import *
from ....lib.Tree import Tree


def test_parent_gets_head():
    t = Tree.from_string('(NP (MISSPLACED-PREDET such) (BNP (DT a) (ADJ perfect) (NN example)) (PP (P of) (PRP it)))')
    bnps = [n for n in t if n.get_label() == 'BNP']
    assert len(bnps) == 1
    assert parent(bnps[0]) == 'NP'


def test_parent_returns_none():
    t = Tree.from_string('(BNP (DT a) (ADJ perfect) (NN example))')
    assert parent(t) is None