from ..extract_features.feature_list import FEATURE_LIST


def _feature_name_to_abbrev(feature_name):
    return ''.join([name_part[0] for name_part in feature_name.split('_')])


def feature_names_to_abbrev(feature_names):
    return [_feature_name_to_abbrev(feature_name) for feature_name in feature_names]


def abbrevs_to_feature_names(abbrev):
    assert len(FEATURE_LIST) == len(set(FEATURE_LIST))
    transl_dict = {
        _feature_name_to_abbrev(feature_name): feature_name for feature_name in FEATURE_LIST
    }
    assert len(transl_dict) == len(FEATURE_LIST)
    return [transl_dict[a] for a in abbrev]
