from ...lib.features import feature_names_to_abbrev, abbrevs_to_feature_names


def test_feature_translate():
    features = [
        'Y_article', 'a_head_form', 'a_head_number', 'a_head_pos', 'a_parent', 'a_non_article_det',
        'a_words_before_head', 'a_pos_before_head', 'a_words_after_head', 'a_pos_after_head',
        'a_words_before_np', 'a_words_after_np', 'a_hypernyms', 'a_referent',
    ]
    abbrevs = [
        'Ya', 'ahf', 'ahn', 'ahp', 'ap', 'anad', 'awbh', 'apbh', 'awah', 'apah', 'awbn', 'awan', 'ah', 'ar'
    ]
    assert abbrevs == feature_names_to_abbrev(features)
    assert features == abbrevs_to_feature_names(abbrevs)

