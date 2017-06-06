def convert_bools(df):
    bool_arrays = {}
    features_to_drop = []
    for feature_name in df.columns:
        if df[feature_name].dtype == bool:
            features_to_drop.append(feature_name)
            bool_arrays[feature_name] = df[feature_name].astype(int).values
    return df.drop(features_to_drop, 1), bool_arrays
