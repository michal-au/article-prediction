import os
import pandas

from ..lib import utils
from .extract import extract_features_from_file, _save_data_frame

SETTINGS = utils.read_settings()


def np_selector(n):
    if n.get_label() != "NP":
        return False
    if not n.children:
        return False
    if "NP" not in [ch.get_label() for ch in n.children]:
        return True
    np_children = [ch for ch in n.children if ch.get_label() == "NP"]
    return all(['POS' in [chch.get_label() for chch in np_child.children] for np_child in np_children])


def main():
    print "START..."

    # EXTRACTING TRAINING FEATURES:
    STARTING_LETTER = 'B'
    path = SETTINGS.get('paths', 'dataBncParsedTrain')
    feature_collection = []  # the resulting feature table
    for r, ds, fs in os.walk(path):
        ds.sort()
        fs.sort()
        fs = [f for f in fs if f[:len(STARTING_LETTER)] == STARTING_LETTER]
        l = len(fs)
        i = 0
        for f in fs:
            i += 1
            f_path = os.path.join(r, f)
            print f_path, i/float(l)
            feature_collection.extend(extract_features_from_file(f_path, np_selector=np_selector, is_test=False))

    df = pandas.DataFrame(feature_collection)
    _save_data_frame(df, 'bnc-train-new-{}'.format(STARTING_LETTER))
    raise NameError

    # EXTRACTING TEST FEATURES:
    path = SETTINGS.get('paths', 'dataBncParsedTest')
    feature_collection = []  # the resulting feature table
    for r, ds, fs in os.walk(path):
        ds.sort()
        fs.sort()
        l = len(fs)
        i = 0
        for f in fs:
            i += 1
            f_path = os.path.join(r, f)
            print f_path, i/float(l)
            feature_collection.extend(extract_features_from_file(f_path, np_selector=np_selector, is_test=False))

    df = pandas.DataFrame(feature_collection)
    _save_data_frame(df, 'bnc-test-new')
    raise NameError


if __name__ == '__main__':
    main()