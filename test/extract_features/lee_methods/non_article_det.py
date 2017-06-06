import pytest

from ....extract_features.lee_methods.feature_methods import *
from ....lib.Tree import Tree


@pytest.mark.parametrize('form', ['That', 'THAT', 'this', 'This', 'another'])
def test_non_article_det_finds(form):
    t = Tree.from_string('(BNP (DT {}) (ADJ perfect) (NN example))'.format(form))
    assert non_article_det(t) == form.lower()


def test_non_article_det_none_with_article():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NN example))')
    assert not non_article_det(t)


def test_non_article_det_none_without_article():
    t = Tree.from_string('(BNP (ADJ perfect) (NN example))')
    assert not non_article_det(t)
