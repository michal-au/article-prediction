from ....extract_features.lee_methods.feature_methods import *
from ....lib.Tree import Tree


def test_words_before_after_np():
    t = Tree.from_string('(R (BNP (DT The) (ADV most) (ADJ perfect) (NNS examples) (PP (P of) (BNP (DT this) (ADJ crazy) (NN stuff)))) (. .))')
    assert words_before_np(t.children[0]) == []
    assert words_after_np(t.children[0]) == ['.']

    bnp = None
    for n in t:
        if n.get_label() == 'BNP' and len(n.children) == 3:
            bnp = n
            break

    assert words_before_np(bnp) == ['example', 'of']
    assert words_after_np(bnp) == ['.']


def test_words_before_after_np2():
    t = Tree.from_string('(R (VP (V See) (BNP (DT The) (ADV most) (ADJ perfect) (NNS examples) (PP (P of) (BNP (DT this) (ADJ crazy) (NN stuff))))))')

    bnp = None
    for n in t:
        if n.get_label() == 'BNP' and len(n.children) == 5:
            bnp = n
            break

    assert words_before_np(bnp) == ['See']
    assert words_after_np(bnp) == []


def test_words_before_np_articles_ignored():
    t = Tree.from_string('(R (BNP (DT The) (ADV most) (ADJ perfect) (NNS examples) (PP (DT the) (BNP (DT this) (ADJ crazy) (NN stuff)))) (. .))')
    bnp = None
    for n in t:
        if n.get_label() == 'BNP' and len(n.children) == 3:
            bnp = n
            break

    assert words_before_np(bnp) == ['example']


def test_words_before_np_nonarticle_dets_not_ignored():
    t = Tree.from_string('(R (BNP (DT The) (ADV most) (ADJ perfect) (NNS examples) (PP (DT this) (BNP (DT this) (ADJ crazy) (NN stuff)))) (. .))')
    bnp = None
    for n in t:
        if n.get_label() == 'BNP' and len(n.children) == 3:
            bnp = n
            break

    assert words_before_np(bnp) == ['example', 'this']


def test_words_after_np_articles_ignored():
    t = Tree.from_string('(R (BNP (DT The) (ADV most) (ADJ perfect) (NNS examples) (PP (P of) (BNP (DT this) (ADJ crazy) (NN stuff)))) (DT an))')
    bnp = None
    for n in t:
        if n.get_label() == 'BNP' and len(n.children) == 3:
            bnp = n
            break

    assert words_after_np(bnp) == []


def test_words_after_np_nonarticle_dets_not_ignored():
    t = Tree.from_string('(R (BNP (DT The) (ADV most) (ADJ perfect) (NNS examples) (PP (P of) (BNP (DT this) (ADJ crazy) (NN stuff)))) (DT this))')
    bnp = None
    for n in t:
        if n.get_label() == 'BNP' and len(n.children) == 3:
            bnp = n
            break

    assert words_after_np(bnp) == ['this']


def test_words_before_np_empty():
    t = Tree.from_string('(BNP (NN example))')
    assert words_before_np(t) == []
    assert words_after_np(t) == []
