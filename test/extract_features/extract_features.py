from collections import deque
import os
import pytest

from ...lib.Tree import Tree
from ...lib.Articles import Article
from ...extract_features.extract import extract_features_from_file, _add_to_history


@pytest.fixture
def input_file(tmpdir):
    f_path = str(tmpdir.join("pos_file"))
    with open(f_path, 'w') as f:
        print >> f, '(R (NPB (DT The) (ADJ perfect) (NN bank)) (. .))'
        print >> f, '(R (NPB (DT The) (ADJ perfect) (NNS dogs)) (. .))'
        print >> f, '(R (NPB (DT The) (ADJ perfect) (NN bank)) (. .))'
    return f_path


def test_add_to_history(input_file):
    with open(input_file, 'r') as f:
        history = deque([None] * 5)
        for l in f:
            t = Tree.from_string(l)
            history = _add_to_history(history, t)

    assert len(history) == 5
    assert len([s for s in history if s]) == 3
    assert not history[0]
    assert history[-1]
    assert history[-2] == ['The', 'perfect', 'dog', '.']


def test_extract_features(tmpdir, input_file):
    assert os.stat(input_file).st_size

    file_features = extract_features_from_file(input_file)
    assert len(file_features) == 3

    references_count = 0
    for sent_features in file_features:
        if sent_features.get('referent'):
            references_count += 1
        assert '_coordinates' in sent_features
        assert '_sent' in sent_features
        assert len(sent_features) == 16

    assert references_count == 1


def test_extract_features_np_selector(tmpdir, input_file):

    def np_selector(x):
        return not x.is_root() and not x.is_leaf() and x.parent.get_label() == 'R' and x.children[0].get_label() == 'DT'
    file_features = extract_features_from_file(input_file, np_selector=np_selector)
    assert len(file_features) == 3

    file_features = extract_features_from_file(input_file, np_selector=lambda x: x.is_root())
    assert len(file_features) == 3

    for sent_features in file_features:
        assert sent_features['parent'] is None
        assert sent_features['article'] == str(Article.ZERO)
