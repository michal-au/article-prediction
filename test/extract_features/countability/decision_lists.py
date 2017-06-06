from __future__ import division
import pytest

from collections import defaultdict
from math import log

from ....extract_features.countability.decision_lists import create_decision_list_from_data
from ....lib.Tree import Tree


def test_create_decision_list_from_data():
    data = {
        'C': [(('a', 'a'), ('b',), ('d',)), (('x',), ('b',), ('d',))],
        'U': [(('a',), ('c',), ('d', 'd'))],
    }
    rules = defaultdict(list)
    for r in create_decision_list_from_data(data):
        rules[r[0]].append((r[1:]))

    assert len(rules) == 3

    assert 'DEFAULT' in rules
    assert rules['DEFAULT'] == [('DEFAULT', 'C', log(2.5/1.5))]

    assert rules['np'] == [
        ('x', 'C', log(1.5/0.5)),
        ('a', 'C', log(2.5/1.5)),
    ]

    assert rules['k+'][0][:2] == ('b', 'C')
    assert abs(rules['k+'][0][2] - log(2.5/0.5)) < 0.000001
    assert rules['k+'][1][:2] == ('c', 'U')
    assert abs(rules['k+'][1][2] - log(1.5/0.5)) < 0.000001
