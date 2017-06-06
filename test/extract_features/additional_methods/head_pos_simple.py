import pytest

from ....extract_features.additional_methods.head_proper import *
from ....lib.Tree import Tree


@pytest.mark.parametrize('tag', ['NN', 'NNP', 'NNS', 'NNPS'])
def test_head_pos_simple(tag):
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) ({} example) )'.format(tag))
    assert head_pos_simple(t) == 'NN'


def test_head_pos_simple_other_tag():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (CD 5) )')
    assert head_pos_simple(t) == 'CD'