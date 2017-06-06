import pytest

from ....extract_features.lee_methods.feature_methods import *
from ....lib.Tree import Tree


def test_head():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NN example) )')
    assert head_form(t) == 'example'
    assert head_pos(t) == 'NN'


def test_head_pos_without_number():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NNS example) )')
    assert head_form(t) == 'example'
    assert head_pos(t) == 'NN'


def test_head_pos_without_number_proper():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NNPS example) )')
    assert head_form(t) == 'example'
    assert head_pos(t) == 'NNP'


def test_head_lemmatizes():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NNS examples))')
    assert head_form(t) == 'example'


# TODO: decide...
#def test_head_lowercases():
#    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NNS Examples))')
#    assert head_form(t) == 'example'


def test_head_converts_numbers():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NNS 42.2) )')
    assert head_form(t) == '<number>'


def test_head_ignores_pps():
    t = Tree.from_string('(BNP (DT the) (ADJ perfect) (NNS example) (POS \'))')
    assert head_form(t) == 'example'


@pytest.mark.parametrize('tag', ['NN', 'NNP', 'NNPS', 'NNS', 'NX', 'JJR'])
def test_head_finds_given_tag_rule2(tag):
    t = Tree.from_string('(BNP (DT the) ({} perfect) ({} example) (PP (P of) (PRP$ mine)))'.format(tag, tag))
    assert head_form(t) == 'example'
    assert head_pos(t) == tag.rstrip('S')


def test_head_finds_given_tag_rule3():
    # this should not happen since we are targeting BNP (they don't dominate NPs
    # by definition)
    t = Tree.from_string('(BNP  (ADJ perfect) (NP example) (NP strange) )')
    assert head_form(t) == 'example'
    assert head_pos(t) == 'NP'


@pytest.mark.parametrize('tag', ['$', 'ADJP', 'PRN'])
def test_head_finds_given_tag_rule4(tag):
    t = Tree.from_string('(BNP (DT the) ({} perfect) ({} example) )'.format(tag, tag))
    assert head_form(t) == 'example'
    assert head_pos(t) == tag


def test_head_finds_given_tag_rule5():
    t = Tree.from_string('(BNP (PRP$ my) (ADJ perfect) (CD 333) )')
    assert head_form(t) == '<number>'
    assert head_pos(t) == 'CD'


@pytest.mark.parametrize('tag', ['JJ', 'JJS', 'RB', 'QP'])
def test_head_finds_given_tag_rule6(tag):
    t = Tree.from_string('(BNP (DT the) ({} perfect) ({} example) )'.format(tag, tag))
    assert head_form(t) == 'example'
    assert head_pos(t) == tag


def test_head_finds_some_creazy_shit():
    t = Tree.from_string('(BNP (DT this) (ADJ creazy) (WTF &^!@#$) )')
    assert head_form(t) == '&^!@#$'
    assert head_pos(t) == 'WTF'


def test_head_returns_none():
    t = Tree.from_string('(R (BNP The))')
    t = t.children[0]
    assert head_form(t) is None
    assert head_pos(t) is None

