FEATURE_SETS = {
    'orig': [
        'a_hypernyms', 'a_head_form', 'a_head_number', 'a_head_pos', 'a_non_article_det', 'a_parent', 'a_referent',
        'b_pos_after_head_as_list', 'b_pos_before_head_as_list', 'b_words_after_head_as_list', 'b_words_after_np_as_list', 'b_words_before_head_as_list', 'b_words_before_np_as_list'
    ],
    'extended': [
        'a_pos_after_head', 'a_pos_before_head', 'a_words_after_head', 'a_words_after_np', 'a_words_before_head', 'a_words_before_np',
        'b_head_proper', 'b_head_pos_simple', 'b_object_form', 'b_pp_object_form', 'b_postmodification_type', 'b_referent',
    ],
    'countability': ['c_countability_bnc'],
    'embeddings': ['d_head_form_embeddings'],
    'lm': ['e_kenlm_ggl_5_lc_nbs'],
}


def get_features_for_set_names(feature_set_combination):
    features = []
    for feature_set in feature_set_combination:
        features.extend(FEATURE_SETS[feature_set])
    # 'a_referent' a 'b_referent' jsou alternativy, nechceme oboje naraz
    # 'a_head_pos' a 'b_head_pos_simple' jsou alternativy, nechceme oboje naraz
    if 'orig' in feature_set_combination and 'extended' in feature_set_combination:
        features.remove('a_referent')
        features.remove('a_head_pos')
    return features
