import pytest

from ....extract_features.countability.countability import countability
from ....extract_features.countability.decision_lists import DEFAULT_NAME
from ....lib.Tree import Tree


def test_countability():
    decision_lists = {'vinken': [
            ('np', 'x', 'C', 5),
            ('np', 'b', 'U', 4),
            ('np', 'c', 'U', 4),
            ('np', 'd', 'C', 4),
            (DEFAULT_NAME, DEFAULT_NAME, 'C', 4),
    ]}
    bnp = Tree.from_string('(NPB (NNP b) (NNP c) (NNP Vinken) )')
    assert countability(bnp, decision_lists, bnp.get_word_tag_pairs()) == 'U'


def test_countability_no_rules():
    decision_lists = {'vinken': [
            ('np', 'x', 'C', 5),
            ('np', 'b', 'U', 4),
            ('np', 'c', 'U', 4),
            ('np', 'd', 'C', 4),
            (DEFAULT_NAME, DEFAULT_NAME, 'C', 4),
    ]}
    bnp = Tree.from_string('(NPB (NNP b) (NNP c) (NNP xxx) )')
    assert countability(bnp, decision_lists, bnp.get_word_tag_pairs()) is None


def test_countability_with_tie():
    decision_lists = {'vinken': [
            ('np', 'x', 'C', 5),
            ('np', 'b', 'C', 4),
            ('np', 'c', 'C', 4),
            ('np', 'd', 'U', 4),
            (DEFAULT_NAME, DEFAULT_NAME, 'U', 4),
    ]}
    bnp = Tree.from_string('(NPB (NNP b) (NNP c) (NNP d) (NNP Vinken) )')
    assert countability(bnp, decision_lists, bnp.get_word_tag_pairs()) is 'U'


def test_countability_with_multiple_ties():
    decision_lists = {'vinken': [
            ('np', 'x', 'C', 5),
            ('np', 'b', 'C', 4),
            ('np', 'c', 'U', 4),
            ('np', 'd', 'C', 3),
            ('np', 'e', 'U', 3),
            ('k+', 'f', 'U', 2),
            (DEFAULT_NAME, DEFAULT_NAME, 'C', 2),
    ]}
    t = Tree.from_string('(TOP (NPB (NNP b) (NNP c) (NNP d) (NNP e) (NNP Vinken) ) (VB (ADJP f)) )')
    bnp = [n for n in t if n.get_label() == 'NPB'][0]
    assert countability(bnp, decision_lists, t.get_word_tag_pairs()) is 'C'


def test_countability_with_multiple_ties_with_intervening_rules():
    decision_lists = {'vinken': [
            ('np', 'x', 'C', 5),
            ('np', 'b', 'C', 4),
            ('np', 'c', 'U', 4),
            ('np', 'xxx', 'U', 3.4),
            ('np', 'xxx', 'U', 3.4),
            ('np', 'xxx', 'U', 3.4),
            ('np', 'd', 'U', 3),
            ('np', 'xxx', 'U', 3),
            ('np', 'e', 'C', 3),
            ('np', 'xxx', 'U', 2.4),
            ('np', 'xxx', 'U', 2.4),
            ('np', 'xxx', 'U', 2.4),
            ('k+', 'f', 'U', 2),
            (DEFAULT_NAME, DEFAULT_NAME, 'C', 2),  # <- this one decides
    ]}
    t = Tree.from_string('(TOP (NPB (NNP b) (NNP c) (NNP d) (NNP e) (NNP Vinken) ) (VB (ADJP f)) )')
    bnp = [n for n in t if n.get_label() == 'NPB'][0]
    assert countability(bnp, decision_lists, t.get_word_tag_pairs()) is 'C'


def test_countability_with_tie_with_single_deciding_rule():
    decision_lists = {'vinken': [
            ('np', 'x', 'C', 5),
            ('np', 'b', 'C', 4),
            ('np', 'c', 'U', 4),
            ('np', 'xxx', 'U', 3.4),
            ('np', 'xxx', 'U', 3.4),
            ('np', 'xxx', 'U', 3.4),
            ('np', 'd', 'U', 3),
            ('np', 'xxx', 'U', 3),
            ('np', 'e', 'C', 3),
            ('np', 'c', 'U', 2.4),  # <- this one decides
            ('k+', 'f', 'U', 2),
            (DEFAULT_NAME, DEFAULT_NAME, 'C', 2),
    ]}
    t = Tree.from_string('(TOP (NPB (NNP b) (NNP c) (NNP d) (NNP e) (NNP Vinken) ) (VB (ADJP f)) )')
    bnp = [n for n in t if n.get_label() == 'NPB'][0]
    assert countability(bnp, decision_lists, t.get_word_tag_pairs()) is 'U'


def test_countability_with_tie_with_single_deciding_rule():
    decision_lists = {'vinken': [
            ('np', 'x', 'C', 5),
            ('np', 'b', 'C', 4),
            ('np', 'c', 'U', 4),
            ('np', 'xxx', 'U', 3.4),
            ('np', 'xxx', 'U', 3.4),
            ('np', 'xxx', 'U', 3.4),
            ('np', 'd', 'U', 3),
            ('np', 'xxx', 'U', 3),
            ('np', 'e', 'C', 3),
            ('np', 'c', 'C', 2.4),  # <-
            ('np', 'xxx', 'U', 2.4),
            ('np', 'c', 'U', 2.4),  # <-
            ('np', 'xxx', 'U', 2.4),
            ('np', 'c', 'C', 2.4),  # <- this one decides
            ('k+', 'f', 'U', 2),
            (DEFAULT_NAME, DEFAULT_NAME, 'C', 2),
    ]}
    t = Tree.from_string('(TOP (NPB (NNP b) (NNP c) (NNP d) (NNP e) (NNP Vinken) ) (VB (ADJP f)) )')
    bnp = [n for n in t if n.get_label() == 'NPB'][0]
    assert countability(bnp, decision_lists, t.get_word_tag_pairs()) is 'C'
