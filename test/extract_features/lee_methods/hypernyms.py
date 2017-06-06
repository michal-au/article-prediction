from ....extract_features.lee_methods.feature_methods import *
from ....lib.Tree import Tree


def test_hypernyms_dog():
    t = Tree.from_string('(R (BNP (DT The) (ADV most) (ADJ perfect) (NN dog) (PP (P of) (BNP (DT the) (ADJ whole) (NN world)))) (. .))')
    bnp = t.children[0]
    assert len(hypernyms(bnp)) == 2


def test_hypernyms_bank():
    t = Tree.from_string('(R (BNP (DT The) (ADJ perfect) (NN bank)) (. .))')
    bnp = t.children[0]
    assert len(hypernyms(bnp)) == 1


def test_hypernyms_empty():
    t = Tree.from_string('(R (BNP (DT The) (ADJ perfect) (NNS ASDFASDFASDFASDF)) (. .))')
    bnp = t.children[0]
    assert hypernyms(bnp) is None
