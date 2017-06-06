import os

from ....lib.utils import read_settings
from ....lib.Tree import Tree


def test_nb_of_training_examples(df_train):
    settings = read_settings()
    path = settings.get("paths", "dataParsed")

    bnp_count = 0
    for r, ds, fs in os.walk(path):
        if r in [os.path.join(path, dir_nb) for dir_nb in ('22', '23', '24')]:
            continue
        for f in fs:
            f_path = os.path.join(r, f)
            with open(f_path, 'r') as parsed_file:
                for l in parsed_file:
                    t = Tree.from_string(l)
                    for n in t:
                        if n.get_label() == 'NPB':
                            bnp_count += 1

    assert len(df_train) == bnp_count


def test_nb_of_heldout_examples(df_heldout, test_forest):
    settings = read_settings()
    path = settings.get("paths", "dataParsed")

    np_count = 0
    for r, ds, fs in os.walk(path):
        if r != os.path.join(path, '22'):
            continue
        for f in fs:
            f_path = os.path.join(r, f)
            with open(f_path, 'r') as parsed_file:
                for i, l in enumerate(parsed_file):
                    t = Tree.from_string(l)
                    for n in t:
                        if n.get_label() == 'NPB':
                            np_count += 1

    assert len(df_heldout) == np_count


def test_nb_of_testing_examples(df_test, test_forest):
    settings = read_settings()
    path = settings.get("paths", "dataParsedOrig")

    np_count = 0
    for r, ds, fs in os.walk(path):
        if r != os.path.join(path, '23'):
            continue
        for f in fs:
            f_path = os.path.join(r, f)
            with open(f_path, 'r') as parsed_file:
                for i, l in enumerate(parsed_file):
                    t = Tree.from_string(l)
                    for n in t:
                        if n.get_label() == 'NP':
                            np_count += 1

    assert len(df_test) < np_count

    non_base_np_count = 0
    for t in test_forest:
        for n in t:
            if n.get_label() == "NP" and "NP" in [ch.get_label() for ch in n.children]:
                np_children = [ch for ch in n.children if ch.get_label() == 'NP']
                if not all(['POS' in [chch.get_label() for chch in np_ch.children] for np_ch in np_children]):
                    non_base_np_count += 1

    assert len(df_test) == np_count - non_base_np_count

    # TODO: neumim vyresit, prebejvaj mi ctyri priklady oproti lee clanku:
    #LEE_COUNTS = 9647 + 324 + 124 + 656 + 1898 + 228 + 167 + 249 + 878
    #assert len(df_test) == LEE_COUNTS