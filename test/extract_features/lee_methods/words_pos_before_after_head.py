from ....extract_features.lee_methods.feature_methods import *
from ....lib.Tree import Tree


def test_words_before_head():
    t = Tree.from_string('(BNP (DT The) (ADV most) (ADJ perfect) (NN example) (PP (P of) (BNP (DT this) (ADJ crazy) (NN stuff))))')
    assert words_before_head(t) == ['most', 'perfect']
    assert pos_before_head(t) == ['ADV', 'ADJ']


def test_words_before_head_empty():
    t = Tree.from_string('(BNP (NN example))')
    assert words_before_head(t) == []
    assert pos_before_head(t) == []


def test_words_before_head_empty_with_article():
    t = Tree.from_string('(BNP (DT The) (NN example))')
    assert words_before_head(t) == []
    assert pos_before_head(t) == []


def test_words_before_head_with_determiner():
    t = Tree.from_string('(BNP (DT Another) (ADJ cool) (NN example))')
    assert words_before_head(t) == ['cool']
    assert pos_before_head(t) == ['ADJ']


def test_words_after_head():
    t = Tree.from_string('(BNP (DT The) (ADV most) (ADJ perfect) (NN example) (PP (P of) (BNP (DT this) (ADJ crazy) (NN stuff))))')
    assert words_after_head(t) == ['of', 'crazy', 'stuff']
    assert pos_after_head(t) == ['P', 'ADJ', 'NN']


def test_words_after_head_empty():
    t = Tree.from_string('(BNP (NN example))')
    assert words_after_head(t) == []
    assert pos_after_head(t) == []
