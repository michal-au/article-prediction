from ....extract_features.additional_methods.np_modification import (
    is_postmodified,
    postmodification_length,
    postmodification_type,
    postmodification_type_specific
)
from ....lib.Tree import Tree


def _get_firts_bnp(t):
    bnp = None
    for n in t:
        if n.get_label() == 'BNP':
            bnp = n
            break
    assert bnp
    return bnp


def test_is_postmodified():
    t = Tree.from_string('(NP (BNP (DT the) (ADJ perfect) (NN example) ) (PP (P of) (BNP (DT that) (NN stuff)) ))')
    assert is_postmodified(_get_firts_bnp(t)) is True


def test_is_postmodified_root_false():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NN example) )')
    assert is_postmodified(_get_firts_bnp(t)) is False


def test_is_postmodified_false():
    t = Tree.from_string(
        '(VP'
        '   (NP (BNP (DT the) (ADJ perfect) (NN example) ) )'
        '   (CONJ and)'
        '   (NP (BNP (DT another) (ADJ worse) (NN example) ) )'
        '   (. .)'
        ')'
    )
    assert is_postmodified(_get_firts_bnp(t)) is False


def test_postmodification_length():
    t = Tree.from_string(
        '(NP'
        '   (BNP (DT the) (ADJ perfect) (NN example) )'
        '   (PP (P of) (BNP (DT that) (NN stuff)))'
        ')'
    )
    assert postmodification_length(_get_firts_bnp(t)) == 3


def test_postmodification_length_other_parent():
    t = Tree.from_string(
        '(VP'
        '   (BNP (DT the) (ADJ perfect) (NN example) )'
        '   (PP (P of) (BNP (DT that) (NN stuff)))'
        ')'
    )
    assert postmodification_length(_get_firts_bnp(t)) == 0


def test_postmodification_length_no_postmod():
    t = Tree.from_string(
        '(NP'
        '   (BNP (DT the) (ADJ perfect) (NN example) )'
        ')'
    )
    assert postmodification_length(_get_firts_bnp(t)) == 0


def test_postmod_type():
    t = Tree.from_string('(NP (BNP (DT the) (ADJ perfect) (NN example) ) (PP (P of) (BNP (DT that) (NN stuff)) ))')
    assert postmodification_type(_get_firts_bnp(t)) == 'PP'


def test_postmod_type_none():
    t = Tree.from_string('(NP (BNP (DT the) (ADJ perfect) (NN example) ))')
    assert postmodification_type(_get_firts_bnp(t)) is None


def test_postmod_type_none_other_parent():
    t = Tree.from_string('(VP (BNP (DT the) (ADJ perfect) (NN example) ) (PP (P of) (BNP (DT that) (NN stuff)) ))')
    assert postmodification_type(_get_firts_bnp(t)) is None


def test_postmod_type_spec():
    t = Tree.from_string('(NP (BNP (DT the) (ADJ perfect) (NN example) ) (PP (P of) (BNP (DT that) (NN stuff)) ))')
    assert postmodification_type_specific(_get_firts_bnp(t)) == 'of'


def test_postmod_type_spec_none():
    t = Tree.from_string('(NP (BNP (DT the) (ADJ perfect) (NN example) ))')
    assert postmodification_type_specific(_get_firts_bnp(t)) is None


def test_postmod_type_spec_none_other_parent():
    t = Tree.from_string('(VP (BNP (DT the) (ADJ perfect) (NN example) ) (PP (P of) (BNP (DT that) (NN stuff)) ))')
    assert postmodification_type_specific(_get_firts_bnp(t)) is None
