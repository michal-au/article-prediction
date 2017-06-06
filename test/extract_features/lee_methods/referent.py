from typing import List

import pytest

from ....extract_features.lee_methods.feature_methods import *
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


def test_referent_true(history):
    assert referent('example', history)
    assert referent('cat', history)


def test_referent_false(history):
    assert not referent('x', history)


def test_referent_false_with_weird_history():
    history = ()
    assert not referent('x', history)

    history = (None, None)
    assert not referent('x', history)

    history = (None, None, ['a'])
    assert not referent('x', history)
