from typing import List
from ....lib.Tree import Tree

import pytest

from ....extract_features.additional_methods.others import extended_referent
from ....lib.wordnet import lemmatize # type: ignore


def lemmatize_tree(t):
    # type (Tree) -> List[str]
    return [lemmatize(word, tag) for (word, tag) in t.get_word_tag_pairs()]

@pytest.fixture
def history():
    return (
        lemmatize_tree(Tree.from_string('(BNP (DT the) (ADJ perfect) (NNS examples) )')),
        lemmatize_tree(Tree.from_string('(BNP (DT a) (ADJ small) (NN cat) )')),
        lemmatize_tree(Tree.from_string('(BNP (DT some) (ADJ small) (NNS cats) )')),
        lemmatize_tree(Tree.from_string('(BNP (DT a) (ADJ small) (NN cat) )')),
        lemmatize_tree(Tree.from_string('(BNP (DT a) (ADJ small) (NN cat) )')),
    )


def test_referent_true_in_history(history):
    t = Tree.from_string('(BNP (DT a) (ADJ smallish) (NN rat) )')
    assert extended_referent('example', t, history)
    assert extended_referent('cat', t, history)


def test_referent_true_in_the_same_sentence(history):
    t = Tree.from_string('(S (BNP (DT some) (ADJ smallish) (NNS rats)) (BNP (DT another) (ADJ smallish) (NN rat)))')
    bnp = t.children[1]
    assert extended_referent('rat', bnp, history)


def test_referent_false(history):
    t = Tree.from_string('(BNP (DT a) (ADJ smallish) (NN rat) )')
    assert not extended_referent('x', t, history)


def test_referent_false_with_weird_history():
    t = Tree.from_string('(BNP (DT a) (ADJ smallish) (NN rat) )')
    history = ()
    assert not extended_referent('x', t, history)

    history = (None, None)
    assert not extended_referent('x', t, history)

    history = (None, None, ['a'])
    assert not extended_referent('x', t, history)
