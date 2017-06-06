import codecs
import pickle
import os

from ..extract_contexts import extract_contexts
from ....lib.utils import read_settings
from ....lib.Tree import Tree


SETTINGS = read_settings()


if __name__ == '__main__':
    path = SETTINGS.get('paths', 'dataBncParsedTrain')
    contexts = {}  # the resulting feature table
    for r, ds, fs in os.walk(path):
        ds.sort()
        fs.sort()
        for f in fs:
            f_path = os.path.join(r, f)
            print f_path
            with codecs.open(f_path, 'r', 'utf-8') as fl:
                for l in fl:
                    if l:
                        tree = Tree.from_string(l)
                        contexts.update(extract_contexts(tree))

    print len(contexts)
    out_path = SETTINGS.get('paths', 'dataCountability')
    pickle_path = os.path.join(out_path, 'contexts-bnc.pkl')
    pickle.dump(contexts, open(pickle_path, 'wb'))
