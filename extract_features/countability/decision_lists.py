from __future__ import division

from math import log
from collections import defaultdict
import pickle
import os

from ...lib.wordnet import lemmatize
from ...lib.utils import read_settings
from ...lib.Tree import Tree


SETTINGS = read_settings()

SMOOTHING_PARAMETER = 0.5
TEMPLATE_NAMES = ('np', 'k+', 'k-')
DEFAULT_NAME = 'DEFAULT'


def create_decision_list_from_data(data):
    # type: Dict[List[Tuple[List]]] -> List[Tuple[str, str, str, float]]
    """
    Input: {
        C: [((words inside np), (words after np), (words before np)), (...)]
        U: [((words inside np), (words after np), (words before np)), (...)]
    }

    Output: [
        ('template (np/k+/k-)', 'word', 'C/U', 'confidence score'),
        (...),
        ...
    ]
    """
    context_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    target_class_counts = defaultdict(float)
    for target_class in data:
        target_class_counts[target_class] = len(data[target_class])
        for occurrence in data[target_class]:
            for i, template in enumerate(occurrence):
                for token in template:
                    context_counts[TEMPLATE_NAMES[i]][token][target_class] += 1

    rules = []
    for template in context_counts:
        for token in context_counts[template]:
            c_counts = context_counts[template][token]['C']
            u_counts = context_counts[template][token]['U']
            sum_ = c_counts + u_counts
            rules.append(
                (template, token, 'C', log(
                    ((c_counts + SMOOTHING_PARAMETER)/(sum_ + 2 * SMOOTHING_PARAMETER)) /
                    ((u_counts + SMOOTHING_PARAMETER)/(sum_ + 2 * SMOOTHING_PARAMETER))
                ))
            )
            rules.append(
                (template, token, 'U', log(
                    ((u_counts + SMOOTHING_PARAMETER)/(sum_ + 2 * SMOOTHING_PARAMETER)) /
                    ((c_counts + SMOOTHING_PARAMETER)/(sum_ + 2 * SMOOTHING_PARAMETER))
                ))
            )

    major_class, minor_class = ('C', 'U') if target_class_counts['C'] > target_class_counts['U'] else ('U', 'C')
    cutoff_value = log(
        (target_class_counts[major_class] + SMOOTHING_PARAMETER) /
        (target_class_counts[minor_class] + SMOOTHING_PARAMETER)
    )

    rules = sorted(
        [r for r in rules if r[3] >= cutoff_value],
        key=lambda x: x[3],
        reverse=True
    )
    rules.append(
        (DEFAULT_NAME, DEFAULT_NAME, major_class, cutoff_value)
    )
    return rules


def extract_decision_lists(data):
    return {word: create_decision_list_from_data(word_data) for word, word_data in data.iteritems()}


if __name__ == '__main__':
    path = SETTINGS.get('paths', 'dataCountability')
    contexts_path = os.path.join(path, 'contexts.pkl')
    lists_path = os.path.join(path, 'decision_lists.pkl')
    data = pickle.load(open(contexts_path, 'r'))

    decision_lists = extract_decision_lists(data)
    pickle.dump(decision_lists, open(lists_path, 'wb'))
