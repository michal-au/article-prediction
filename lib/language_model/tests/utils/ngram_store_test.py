import pytest
from utils.ngram_store import NgramStore


def test_dynamic_defaultdict():
    store = NgramStore(3)
    assert store.forward_store['a']['b']['c'] == 0

    store = NgramStore(1)
    assert store.forward_store['a'] == 0

# def test_adding():
#     store = NgramStore(5)
#     store.add([
#         ('a', 'b', 'c', 'd', 'd'),
#         # ('a', 'b', 'c', 'd', 'd'),
#         # ('a', 'b', 'c', 'd', 'e'),
#         # ('a', 'b', 'c', 'd', 'f'),
#     ])
#     assert store.total_count == 4
