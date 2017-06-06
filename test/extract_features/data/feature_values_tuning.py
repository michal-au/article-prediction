from .conftest import is_nonempty, is_boolean


def test_feature_data_head_pos_simple_no_empty(df_train_tun, df_test_tun, df_heldout_tun):
    for df in (df_train_tun, df_test_tun, df_heldout_tun):
        assert is_nonempty(df['b_head_pos_simple'])


def test_feature_data_head_proper(df_train_tun, df_test_tun, df_heldout_tun):
    for df in (df_train_tun, df_test_tun, df_heldout_tun):
        assert is_boolean(df['b_head_proper'])
