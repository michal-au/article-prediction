from collections import deque
import os
import pytest

from ...lib.Tree import Tree
from ...extract_features.np_selector import penn_selector


def test_selector_non_np():
    t = Tree.from_string('(R (NP (DT The) (ADJ perfect) (NN bank)) (. .))')
    assert not penn_selector(t)


def test_selector_pos_parent():
    t = Tree.from_string('(NP (NP (DT the) (ADJ perfect) (NN bank) (POS \'s)) (NN loan))')
    assert penn_selector(t)


def test_selector_pos_parent2():
    t = Tree.from_string('(NP (DT the) (NP (NNP Bigg) (POS \'s)) (NN hypermarket))')
    assert penn_selector(t)


def test_selector_pos():
    t = Tree.from_string('(NP (NP (DT the) (ADJ perfect) (NN bank) (POS \'s)) (NN loan))')
    assert penn_selector(t.children[0])
