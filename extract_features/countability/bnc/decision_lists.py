from __future__ import division

import pickle
import os

from ....lib.utils import read_settings
from ..decision_lists import create_decision_list_from_data

SETTINGS = read_settings()


def extract_decision_lists(data):
    rules = {}
    data_length = len(data)
    counter = 0
    for word, word_data in data.iteritems():
        counter += 1
        print "{} out of {}".format(counter, data_length)
        rules[word] = create_decision_list_from_data(word_data)
    return rules


if __name__ == '__main__':
    path = SETTINGS.get('paths', 'dataCountability')
    contexts_path = os.path.join(path, 'contexts-bnc.pkl')
    lists_path = os.path.join(path, 'decision_lists-bnc.pkl')
    data = pickle.load(open(contexts_path, 'r'))

    decision_lists = extract_decision_lists(data)
    pickle.dump(decision_lists, open(lists_path, 'wb'))
