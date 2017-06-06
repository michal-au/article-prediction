from ....extract_features.list_to_binary_features import resolve_list_features, LIST_FEATURES
LIST_FEATURES = [lf[0] for lf in LIST_FEATURES]


def test_resolve_features(df_simple):
    df_list = resolve_list_features(df_simple)

    new_features = {}
    for example in df_list:
        for f in example:
            assert not isinstance(f, list)
            new_features[f] = True

    feature_count = len(df_simple.columns) - len(LIST_FEATURES)
    for f in df_simple.columns.values:
        if f not in LIST_FEATURES:
            continue
        feature_count += len(set([
            val for r in df_simple[f] if r
            for val in r
        ]))
    assert len(new_features) == feature_count
