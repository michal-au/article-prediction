from __future__ import division
from ....extract_features.additional_methods.others import relative_position_within_sent, main_verb_form
from ....lib.Tree import Tree


def _get_first_bnp(t):
    bnp = None
    for n in t:
        if n.get_label() == 'NPB':
            bnp = n
            break
    assert bnp
    return bnp


def test_relative_position():
    t = Tree.from_string('(VP (NPB (DT the) (ADJ perfect) (NN example) ) (PP (P of) (BNP (DT that) (NN stuff)) ))')
    assert relative_position_within_sent(_get_first_bnp(t)) == 0

    t = Tree.from_string('(VP (RANDOM Wov) (RANDOM Wov) (NPB (DT the) (ADJ perfect) (NN example) ) (PP (P of) (BNP (DT that) (NN stuff)) ))')
    assert relative_position_within_sent(_get_first_bnp(t)) == 2/6
