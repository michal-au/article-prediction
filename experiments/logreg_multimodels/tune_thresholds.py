import numpy as np
from code.lib.utils import load_postprocessed_feature_data

FEATURES = ['a_hypernyms', 'a_head_form', 'a_head_number', 'a_non_article_det', 'a_parent', 'a_pos_after_head', 'a_pos_before_head', 'a_words_after_head', 'a_words_after_np', 'a_words_before_head', 'a_words_before_np', 'b_head_proper', 'b_head_pos_simple', 'b_object_form', 'b_pos_after_head_as_list', 'b_pos_before_head_as_list', 'b_pp_object_form', 'b_postmodification_type', 'b_referent', 'b_words_after_head_as_list', 'b_words_after_np_as_list', 'b_words_before_head_as_list', 'b_words_before_np_as_list', 'c_countability_bnc', 'd_head_form_embeddings', 'e_kenlm_ggl_5_lc_nbs']
TRAIN_SET_NAME = 'train'
TEST_SET_NAME = 'heldout'
TRAINING_DATA_SIZE = 20000


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def cost(Y, T):
    # Y - predicted matrix
    # T - target matrix
    return - np.multiply(T, np.log(Y)).sum()

if __name__ == '__main__':
    data = {}
    data['train_x'], data['train_y'] = load_postprocessed_feature_data(FEATURES, TRAIN_SET_NAME, sample=TRAINING_DATA_SIZE)
    data['test_x'], data['test_y'] = load_postprocessed_feature_data(FEATURES, TEST_SET_NAME)
    assert data['train_x'].shape[1] == data['test_x'].shape[1]
