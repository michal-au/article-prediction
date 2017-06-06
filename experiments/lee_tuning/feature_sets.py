
basic = [
    'a_head_form',
    'a_head_number',
    'a_head_pos',
    'a_hypernyms',
    'a_non_article_det',
    'a_parent',
    'a_pos_after_head',
    'a_pos_before_head',
    'a_referent',
    'a_words_after_head',
    'a_words_after_np',
    'a_words_before_head',
    'a_words_before_np',
]

basic_modified = [
    'a_head_form',
    'a_head_number',
    'a_head_pos',
    'a_hypernyms',
    'a_non_article_det',
    'a_parent',
    'b_pos_after_head_as_list',
    'b_pos_before_head_as_list',
    'a_referent',
    'b_words_after_head_as_list',
    'b_words_after_np_as_list',
    'b_words_before_head_as_list',
    'b_words_before_np_as_list',
]

basic_modified_v2 = [
    'a_head_form',
    'a_head_number',
    'a_head_pos',
    'a_hypernyms',
    'a_non_article_det',
    'a_parent',
    'b_pos_after_head_as_list',
    'b_pos_before_head_as_list',
    'a_referent',
    'b_words_after_head_as_list',
    'b_words_after_np_as_list',
    'b_words_before_head_as_list',
    'b_words_before_np_as_list',
    'a_pos_after_head',
    'a_pos_before_head',
    'a_words_after_head',
    'a_words_after_np',
    'a_words_before_head',
    'a_words_before_np',
]

basic_modified_v3 = [
    'a_head_form',
    'a_head_number',
    'a_head_pos',
    'a_hypernyms',
    'a_non_article_det',
    'b_parent',
    'b_pos_after_head_as_list',
    'b_pos_before_head_as_list',
    'a_referent',
    'b_words_after_head_as_list',
    'b_words_after_np_as_list',
    'b_words_before_head_as_list',
    'b_words_before_np_as_list',
    'a_pos_after_head',
    'a_pos_before_head',
    'a_words_after_head',
    'a_words_after_np',
    'a_words_before_head',
    'a_words_before_np',
]

basic_modified_v4 = [
    'a_head_form',
    'a_head_number',
    'a_head_pos',
    'a_hypernyms',
    'a_non_article_det',
    'b_parent',
    'b_pos_after_head_as_list',
    'b_pos_before_head_as_list',
    'b_referent',
    'b_words_after_head_as_list',
    'b_words_after_np_as_list',
    'b_words_before_head_as_list',
    'b_words_before_np_as_list',
    'a_pos_after_head',
    'a_pos_before_head',
    'a_words_after_head',
    'a_words_after_np',
    'a_words_before_head',
    'a_words_before_np',
]

basic_modified_v5 = [
    'a_head_form',
    'a_head_number',
    'a_head_pos',
    'a_hypernyms',
    'a_non_article_det',
    'b_parent',
    'b_pos_after_head_as_list',
    'b_pos_before_head_as_list',
    'b_referent_with_propers',
    'b_words_after_head_as_list',
    'b_words_after_np_as_list',
    'b_words_before_head_as_list',
    'b_words_before_np_as_list',
    'a_pos_after_head',
    'a_pos_before_head',
    'a_words_after_head',
    'a_words_after_np',
    'a_words_before_head',
    'a_words_before_np',
]

b_features = [
    'a_head_form',
    'a_head_number',
    'b_head_pos_simple',
    'b_head_proper',
    'a_hypernyms',
    'a_non_article_det',
    'b_parent',
    'b_pos_after_head_as_list',
    'b_pos_before_head_as_list',
    'b_referent',
    'b_words_after_head_as_list',
    'b_words_after_np_as_list',
    'b_words_before_head_as_list',
    'b_words_before_np_as_list',
    'a_pos_after_head',
    'a_pos_before_head',
    'a_words_after_head',
    'a_words_after_np',
    'a_words_before_head',
    'a_words_before_np',
    'b_postmodification_type',
    'b_object_form',
    'b_pp_object_form',
]


b_c_features = [
    'a_head_form',
    'a_head_number',
    'b_head_pos_simple',
    'b_head_proper',
    'a_hypernyms',
    'a_non_article_det',
    'b_parent',
    'b_pos_after_head_as_list',
    'b_pos_before_head_as_list',
    'b_referent',
    'b_words_after_head_as_list',
    'b_words_after_np_as_list',
    'b_words_before_head_as_list',
    'b_words_before_np_as_list',
    'a_pos_after_head',
    'a_pos_before_head',
    'a_words_after_head',
    'a_words_after_np',
    'a_words_before_head',
    'a_words_before_np',
    'b_postmodification_type',
    'b_object_form',
    'b_pp_object_form',
    'c_countability_bnc',
]

b_c_features_v2 = b_c_features + ['c_referent_with_countability']

b_c_features_v3 = b_c_features + ['c_referent_with_countability_with_propers']

b_c_d_features = b_c_features + ['d_head_form_embeddings']

b_c_d_features_v2 = [
    'a_head_form',  # 0
    'a_head_number',  # 1
    'b_head_pos_simple',  # 13
    'b_head_proper',  # 14
    'a_hypernyms',  # 3
    'a_non_article_det',  # 4
    'b_parent',  # 24
    'b_pos_after_head_as_list',
    'b_pos_before_head_as_list',
    'b_referent',  # 21
    'b_words_after_head_as_list',
    'b_words_after_np_as_list',
    'b_words_before_head_as_list',
    'b_words_before_np_as_list',
    'b_postmodification_type',  # 18
    'b_object_form',  # 25
    'b_pp_object_form',  # 27
    'c_countability_bnc',  # 44
    'd_head_form_embeddings',  # 29
]


features = [
    'a_head_form',
    'a_head_number',
    #'a_head_pos',
    'a_hypernyms',
    'a_non_article_det',
    #'a_parent',
    'a_pos_after_head',
    'a_pos_before_head',
    #'a_referent',
    'a_words_after_head',
    'a_words_after_np',
    'a_words_before_head',
    'a_words_before_np',
    'b_head_pos_simple',
    'b_head_proper',
    #'b_is_postmodified',
    #'b_postmodification_length',
    #'b_postmodification_pp_of',
    'b_postmodification_type',
    #'b_postmodification_type_specific',
    #'b_non_article_det_extended',
    #'b_relative_position',
    #'b_predicate_form',
    'b_referent',
    #'b_subject_predicate',
    #'b_subject_predicate2',  # todo: test this shit!!! see feature extraction
    #'b_subject_position',
    'b_parent',
    'b_object_form',  # todo rename
    #'b_object_form_cutoff',   # todo: test this shit!!! see feature extraction
    'b_pp_object_form',
    #'b_position_within_article',
    'd_head_form_embeddings',
    #'d_object_form_embeddings',
    #'d_words_before_head_embeddings',
    #'d_words_after_head_embeddings',
    #'d_words_before_np_embeddings',
    #'d_words_after_np_embeddings',
    'c_countability_bnc',
]