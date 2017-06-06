import pytest

from ....extract_features.lee_methods.feature_methods import *
from ....lib.Tree import Tree


@pytest.mark.parametrize('tag', ['NN', 'NNP'])
def test_number_sg(tag):
    t = Tree.from_string('(BNP (DT a) (ADJ perfect) ({} example))'.format(tag))
    assert head_number(t) == 'sg'


@pytest.mark.parametrize('tag', ['NNS', 'NNPS'])
def test_number_pl(tag):
    t = Tree.from_string('(BNP (DT a) (ADJ perfect) ({} example))'.format(tag))
    assert head_number(t) == 'pl'


def test_number_unknown():
    t = Tree.from_string('(BNP (DT a) (ADJ perfect) (WTF example))')
    assert not head_number(t)
