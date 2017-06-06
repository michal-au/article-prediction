import pytest
from utils.ngrams import ngrams
from utils.constants import Constants


def test_basic():
    assert ngrams(['a'], 1) == [('a',)]
    assert ngrams(['a', 'b'], 1) == [('a',), ('b',)]
    assert ngrams(['a', 'b'], 2) == [('a', 'b')]
    assert ngrams(['a', 'b', 'c', 'd', 'e'], 2) == [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')]
    assert ngrams(['a'], 5) == []


def test_padding():
    c = Constants.fake_token
    assert ngrams(['a'], 4, padding=True) == [(c, c, c, 'a'), (c, c, 'a', c), (c, 'a', c, c), ('a', c, c, c)]
    assert ngrams(['a'], 2, padding=True) == [(c, 'a'), ('a', c)]
