from .train_model import train_model, grid_search


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
'b_subject_predicate',
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


if __name__ == '__main__':
    call = "LogisticRegression(solver='lbfgs', multi_class='multinomial', C=0.1)"
    #call = "SGDClassifier(loss='log', penalty='l2', n_iter=15, alpha=0.0001)"
    train_model(call)

    #cross_valid_call = "SGDClassifier(random_state=423, n_jobs=-1)"
    #grid_search(cross_valid_call)

