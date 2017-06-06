from ....extract_features.additional_methods.head_proper import head_proper
from ....lib.Tree import Tree


def test_head_proper_NN():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NN example) )')
    assert head_proper(t) is False


def test_head_proper_NNS():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NNS example) )')
    assert head_proper(t) is False


def test_head_proper_NNP():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NNP example) )')
    assert head_proper(t) is True


def test_head_proper_NNPS():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NNPS example) )')
    assert head_proper(t) is True



