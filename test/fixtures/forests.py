import os
import pytest

from ...lib.utils import read_settings
from ...lib.Tree import Tree

__all__ = ('test_forest',)


@pytest.fixture
def test_forest():
    settings = read_settings()
    path = settings.get("paths", "dataParsedOrig")

    forest = []

    for f_name in sorted(os.listdir(os.path.join(path, '23'))):
        with open(os.path.join(path, '23', f_name), 'r') as f:
            for l in f:
                forest.append(Tree.from_string(l))
    assert len(forest) == 2416  # find data/parsed_orig/23 -name '*.mrg' | xargs wc -l
    return forest
