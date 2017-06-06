import pytest

from ..Tree import Tree


def test_from_string():
    t = Tree.from_string('(R (L1 l1) (L2 l2))')
    assert [n.val for n in t] == ['R', 'L1', 'l1', 'L2', 'l2']


def test_from_string():
    t = Tree.from_string('(R (L1 l1 l1 l1) (L2 l2))')
    assert [n.val for n in t] == ['R', 'L1', 'l1 l1 l1', 'L2', 'l2']


@pytest.fixture
def tree():
    return Tree.from_string('(R (L1 (L2 (L3 l3))) (L4 l4) )')


def test_to_string(tree):
    assert tree.to_string() == ' (R (L1 (L2 (L3 l3))) (L4 l4))'


def test_delete_node(tree):
    assert not tree.delete_node()
    assert tree.children[0].children[0].delete_node()
    assert tree.to_string() == ' (R (L1) (L4 l4))'


def test_delete_nodes_by_value(tree):
    tree.delete_nodes_by_value('L3')
    assert tree.to_string() == ' (R (L4 l4))'

    tree = Tree.from_string('(R (L1 l1) (L2 (L3 (L1 l1))) (L4 l4))')
    tree.delete_nodes_by_value('L1')
    assert tree.to_string() == ' (R (L4 l4))'

    with pytest.raises(NameError):
        tree.delete_nodes_by_value('L4')


def test_get_root(tree):
    assert tree.get_root().get_label() == 'R'
    assert tree.children[0].get_root().get_label() == 'R'  # L1
    assert tree.children[0].children[0].get_root().get_label() == 'R'  # L2
    assert tree.children[0].children[0].children[0].get_root().get_label() == 'R'  # L3
    assert tree.children[1].get_root().get_label() == 'R'  # L4


def test_get_leftmost_child(tree):
    assert tree.get_leftmost_child().get_label() == 'L3'  # R
    assert tree.children[0].get_leftmost_child().get_label() == 'L3'  # L1
    assert tree.children[0].children[0].get_leftmost_child().get_label() == 'L3'  # L2
    assert tree.children[0].children[0].children[0].get_leftmost_child().get_label() == 'L3'  # L3
    assert tree.children[1].get_leftmost_child().get_label() == 'L4'  # L4


def test_get_rightmost_child(tree):
    assert tree.get_rightmost_child().get_label() == 'L4'  # R
    assert tree.children[0].get_rightmost_child().get_label() == 'L3'  # L1
    assert tree.children[0].children[0].get_rightmost_child().get_label() == 'L3'  # L2
    assert tree.children[0].children[0].children[0].get_rightmost_child().get_label() == 'L3'  # L3
    assert tree.children[1].get_rightmost_child().get_label() == 'L4'  # L4


def test_iteration(tree):
    tree = Tree.from_string('(R (L1 (L2 (L3 l3))) )')
    assert len([True for n in tree]) == 5
    assert len([True for n in tree for nn in n]) == 15


def test_insert_article():
    tree = Tree.from_string('(R (L1 (L2 (L3 l3)) (L4 (L5 l5))) )')
    n = [n for n in tree if n.get_label() == 'L4'][0]
    n.insert_article('the')
    assert tree.children[0].children[1].children[0].children[0].get_label() == 'the'  # R


def test_insert_article_start_of_sentence():
    tree = Tree.from_string('(R (L1 (L2 (L3 l3)) (L4 l4)) )')
    n = [n for n in tree if n.get_label() == 'L1'][0]
    n.insert_article('the')
    assert tree.get_leftmost_child().children[0].get_label() == 'The'  # R


def test_insert_article_upper_case():
    tree = Tree.from_string('(R (L1 (L2 (L3 ASDF)) (L4 KVOK)) )')
    n = [n for n in tree if n.get_label() == 'L1'][0]
    n.insert_article('the')
    assert tree.get_leftmost_child().children[0].get_label() == 'THE'  # R
