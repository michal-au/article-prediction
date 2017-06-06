from __future__ import division

from collections import defaultdict
import numpy as np
import pandas


LIST_FEATURES = (
    ('a_hypernyms', 'h'),
    ('a_pos_after_head', 'pah'),
    ('a_pos_before_head', 'pbh'),
    ('a_words_after_head', 'wah'),
    ('a_words_after_np', 'wan'),
    ('a_words_before_head', 'wbh'),
    ('a_words_before_np', 'wbn')
)


ALPHA = 0.5


def convert_list_vector(df, vector_name, feature_value_counts):
    counts = feature_value_counts[vector_name]

    for target_class in ('THE', 'A', 'ZERO'):
        result_list = []
        for example in df[vector_name]:
            if not example:
                result_list.append(None)
                continue
            assert isinstance(example, list), example
            example_words_total_count = sum([sum(counts[word].values()) for word in example])
            if example_words_total_count:
                result_list.append(
                    sum([
                        _count_probability(word, target_class, counts) * sum(counts[word].values()) for word in example
                    ]) / example_words_total_count
                )
            else:
                result_list.append(None)
        df[vector_name + '_' + target_class] = pandas.Series(result_list)

    df = df.drop(vector_name, 1)
    return df


def convert_nominal_vector(df, vector_name, feature_value_counts):
    counts = feature_value_counts[vector_name]

    for target_class in ('THE', 'A', 'ZERO'):
        df[vector_name + '_' + target_class] = pandas.Series(
            [_count_probability(value, target_class, counts) for value in df[vector_name]]
        )

    df = df.drop(vector_name, 1)
    return df


def get_feature_value_counts(df, cutoff, feature_names):
    list_feature_names = [f[0] for f in LIST_FEATURES]
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for row in df.to_dict(orient='records'):
        for feature_name in feature_names:
            if feature_name in list_feature_names:
                if row[feature_name]:
                    for word in row.get(feature_name, []):
                        counts[feature_name][word][row['Y_article']] += 1
            else:
                counts[feature_name][row[feature_name]][row['Y_article']] += 1

    counts = _handle_oovs(counts, cutoff)
    return counts


def _count_probability(value, target_class, counts):
    if value in counts:
        return (counts[value][target_class] + ALPHA) / (sum(counts[value].values()) + 3 * ALPHA)
    else:
        return (counts['-OOV-'][target_class] + ALPHA) / (sum(counts['-OOV-'].values()) + 3 * ALPHA)


def _handle_oovs(counts, cutoff):
    for feature in counts.keys():
        for token in counts[feature].keys():
            if sum(counts[feature][token].values()) < cutoff:
                for target_class, count in counts[feature][token].iteritems():
                    counts[feature]['-OOV-'][target_class] += count
                del counts[feature][token]
    return counts
