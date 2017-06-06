from ....lib.corpus import DataType, walk_parses
from ....lib.Tree import Tree


def _has_parent(t):
    # (Tree) -> None
    for n in t:
        if n.get_label() == 'NPB':
            assert n.parent


def test_bnps_have_parent():
    walk_parses(_has_parent, data_type=DataType.ALL)
