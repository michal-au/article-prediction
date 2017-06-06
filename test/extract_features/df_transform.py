from collections import defaultdict
import pandas
import numpy
import pytest

from ...extract_features.df_transform import _handle_oovs, convert_nominal_vector, convert_list_vector, get_feature_value_counts
from ...extract_features.list_to_binary_features import prepare_for_vectorizer


def test_handle_oovs():
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    counts['vector']['a']['A'] = 10
    counts['vector']['a']['B'] = 10
    counts['vector']['a']['C'] = 10

    counts['vector']['b']['A'] = 10
    counts['vector']['b']['B'] = 10
    counts['vector']['b']['C'] = 10

    counts['vector']['c']['A'] = 1
    counts['vector']['c']['B'] = 2

    counts['vector']['d']['A'] = 2
    counts['vector']['d']['C'] = 1
    counts = _handle_oovs(counts, 5)

    assert counts == {
        'vector': {
            'a': {
                'A': 10,
                'B': 10,
                'C': 10,
            },
            'b': {
                'A': 10,
                'B': 10,
                'C': 10,
            },
            '-OOV-': {
                'A': 3,
                'B': 2,
                'C': 1,
            },
        }
    }


def test_convert_nominal_vector():
    rows = [
        ('a', 'THE'),
        ('a', 'THE'),
        ('a', 'THE'),
        ('a', 'THE'),
        ('a', 'THE'),
        ('a', 'A'),
        ('a', 'A'),
        ('a', 'A'),
        ('a', 'ZERO'),
        ('a', 'ZERO'),
        ('b', 'A'),
        ('b', 'A'),
        ('b', 'A'),
        ('b', 'ZERO'),
        ('b', 'ZERO'),
    ]
    df = pandas.DataFrame(rows, columns=['vector', 'Y_article'])
    feature_value_counts = get_feature_value_counts(df, 5, ['vector'])
    df = convert_nominal_vector(df, 'vector', feature_value_counts)
    assert list(df['vector_THE']) == [5.5/11.5] * 10 + [0.5/6.5] * 5
    assert list(df['vector_A']) == [3.5/11.5] * 10 + [3.5/6.5] * 5
    assert list(df['vector_ZERO']) == [2.5/11.5] * 10 + [2.5/6.5] * 5
    assert set(df.columns) == {'Y_article', 'vector_THE', 'vector_A', 'vector_ZERO'}


def test_convert_list_vector():
    vector_name = 'a_hypernyms'
    cutoff = 3
    rows = [
        (['a', 'b', 'c'], 'THE'),
        (['a', 'b', 'c'], 'THE'),
        (['a', 'b'], 'THE'),
        (['a', 'b'], 'THE'),
        (['c', 'd'], 'A'),
        (['c', 'd'], 'A'),
        (['e', 'f'], 'ZERO'),
        (['e', 'f'], 'ZERO'),
    ]
    df = pandas.DataFrame(rows, columns=[vector_name, 'Y_article'])
    feature_value_counts = get_feature_value_counts(df, cutoff, [vector_name])
    df = convert_list_vector(df, vector_name, feature_value_counts)

    assert list(df[vector_name + '_THE'])[:2] == [(4*4.5/5.5 + 4*4.5/5.5 + 4*2.5/5.5)/12] * 2
    assert list(df[vector_name + '_THE'])[2:4] == [(4*4.5/5.5 + 4*4.5/5.5)/8] * 2
    assert list(df[vector_name + '_THE'])[4:6] == [(4*2.5/5.5)/4] * 2
    assert all(numpy.isnan(df[vector_name + '_THE'][6:]))

    assert list(df[vector_name + '_A'])[:2] == [(4*0.5/5.5 + 4*0.5/5.5 + 4*2.5/5.5)/12] * 2
    assert list(df[vector_name + '_A'])[2:4] == [(4*0.5/5.5 + 4*0.5/5.5)/8] * 2
    assert list(df[vector_name + '_A'])[4:6] == [(4*2.5/5.5)/4] * 2
    assert all(numpy.isnan(df[vector_name + '_A'][6:]))

    assert list(df[vector_name + '_ZERO'])[:2] == [(4*0.5/5.5 + 4*0.5/5.5 + 4*0.5/5.5)/12] * 2
    assert list(df[vector_name + '_ZERO'])[2:4] == [(4*0.5/5.5 + 4*0.5/5.5)/8] * 2
    assert list(df[vector_name + '_ZERO'])[4:6] == [(4*0.5/5.5)/4] * 2
    assert all(numpy.isnan(df[vector_name + '_ZERO'][6:]))

    assert set(df.columns) == {'Y_article', vector_name + '_THE', vector_name + '_A', vector_name + '_ZERO'}


def test_prepare_for_vectorizer():
    data = [
        {'a': 0, 'b': True, 'c': [], 'd': ['a', 'b', 'c'], 'e': 'd'},
        {'a': 1, 'b': False, 'c': [1,2,3], 'd': ['a', 'c'], 'e': 'e'},
        {'a': 1, 'b': False, 'c': [1,2,3], 'd': ['b', 'c'], 'e': None},
    ]
    df = pandas.DataFrame(data=data)
    print df

    df, dict_data = prepare_for_vectorizer(df)
    assert set(df.columns) == {'a', 'b', 'c'}

    old_df = pandas.DataFrame(data=data)
    assert list(df.a) == list(old_df.a)
    assert list(df.b) == list(old_df.b)
    assert list(df.c) == list(old_df.c)

    assert len(dict_data) == 3
    assert dict_data[0] == {'d_a': True, 'd_b': True, 'd_c': True, 'e': 'd'}
    assert dict_data[1] == {'d_a': True, 'd_c': True, 'e': 'e'}
    assert dict_data[2] == {'d_b': True, 'd_c': True}
