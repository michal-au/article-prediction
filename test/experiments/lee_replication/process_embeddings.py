import numpy
import pandas
from ....experiments.df_transform.embeddings import process_embeddings


def test_process_embeddings2(df_simple):
    test_vector = []
    for _ in xrange(100):
        test_vector.append(numpy.array([1, 2, 3]))

    df_simple['d_embeddings'] = pandas.Series(test_vector, index=df_simple.index)
    orig_column_count = len(df_simple.columns)

    df = process_embeddings(df_simple)

    assert len(df.columns) == orig_column_count + 2
    assert list(df.d_embeddings_0) == [1.0] * 100
    assert list(df.d_embeddings_1) == [2.0] * 100
    assert list(df.d_embeddings_2) == [3.0] * 100


def test_process_embeddings_with_empty_cells(df_simple):
    test_vector = [None]
    for _ in xrange(99):
        test_vector.append(numpy.array([1, 2, 3]))

    df_simple['d_embeddings'] = pandas.Series(test_vector, index=df_simple.index)
    orig_column_count = len(df_simple.columns)

    df = process_embeddings(df_simple)
    assert len(df.columns) == orig_column_count + 2
    assert df.d_embeddings_0[0] is None
    assert list(df.d_embeddings_0[1:]) == [1.0] * 99
    assert df.d_embeddings_1[0] is None
    assert list(df.d_embeddings_1[1:]) == [2.0] * 99
    assert df.d_embeddings_2[0] is None
    assert list(df.d_embeddings_2[1:]) == [3.0] * 99
