from ....extract_features.lee_methods.feature_methods import *
from ....lib.Tree import Tree
from ....lib.Articles import Article


def test_article_finds_definite():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NN example) )')
    assert article_class(t) == Article.THE


def test_article_finds_indefinite():
    t = Tree.from_string('(BNP (DT a) (ADJ perfect) (NN example) )')
    assert article_class(t) == Article.A


def test_article_finds_zero():
    t = Tree.from_string('(BNP (ADJ perfect) (NNS examples) )')
    assert article_class(t) == Article.ZERO


def test_article_finds_ignores_other_DT():
    t = Tree.from_string('(BNP (DT that) (ADJ perfect) (NN example) )')
    assert article_class(t) == Article.ZERO


def test_article_ignores_nested_NPs():
    t = Tree.from_string('(BNP (ADJ perfect) (NNS subexamples) (PP (P of) (BNP (DT the) (NN example))) )')
    assert article_class(t) == Article.ZERO
