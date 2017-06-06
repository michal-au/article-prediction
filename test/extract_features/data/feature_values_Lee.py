from .conftest import is_boolean, is_nonempty, has_only_values


def test_feature_data_article_values(df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
    for df in (df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
        assert has_only_values(df['Y_article'], {'THE', 'A', 'ZERO'})


def test_feature_data_coordinates_no_empty(df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
    for df in (df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
        assert is_nonempty(df['_coordinates'])


def test_feature_data_head_form_no_empty(df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
    for df in (df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
        assert is_nonempty(df['a_head_form'])


def test_feature_data_head_pos_no_empty(df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
    for df in (df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
        assert is_nonempty(df['a_head_pos'])


def test_feature_data_number_values(df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
    for df in (df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
        assert has_only_values(df['a_head_number'], {'sg', 'pl', None})


def test_feature_data_referent_values(df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
    for df in (df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
        assert is_boolean(df['a_referent'])


def test_feature_data_paired_list_match(df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
    get_len = lambda x: len(x)
    for df in (df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
        for a, b in (('a_words_before_head', 'a_pos_before_head'), ('a_words_after_head', 'a_pos_after_head')):
            assert all(df[a].apply(get_len) == df[b].apply(get_len))


def test_no_feature_always_falsy(df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
    for df in (df_train, df_test, df_heldout, df_train_tun, df_test_tun, df_heldout_tun):
        for col in df:
            assert any(df[col])