import pytest

from ..Constants import Constants
from ..sentence import SentenceStateTracker
from ..Tree import Tree


def test_sentence_state_tracker_zero_filled():
    t = Tree.from_string('(R (L1 l1) (L2 (L3 (L4 l4))) (L5 l5))')
    tokens = [Constants.no_article_token]  + ' {} '.format(Constants.no_article_token).join(t.get_words()).split(' ')
    stt = SentenceStateTracker(tokens, zero_filled=True)

    bnp = next((n for n in t if n.get_label() == 'L2'))
    prediction_candidates = stt.get_prediction_candidates(bnp)
    assert prediction_candidates[0] == '{zero} l1 {indef} l4 {zero} l5'.format(
        zero=Constants.no_article_token, indef=Constants.article_indefinite_bucket_token
    )
    assert prediction_candidates[1] =='{zero} l1 {defi} l4 {zero} l5'.format(
        zero=Constants.no_article_token, defi=Constants.article_definite_token
    )
    assert prediction_candidates[2] == '{zero} l1 {zero} l4 {zero} l5'.format(
        zero=Constants.no_article_token
    )

    stt.record_prediction(bnp, Constants.article_definite_token)
    assert stt.last_prediction_idx == 3
    assert stt.so_far_predicted == [Constants.no_article_token, 'l1', Constants.article_definite_token]


    bnp = next((n for n in t if n.get_label() == 'L5'))
    prediction_candidates = stt.get_prediction_candidates(bnp)
    assert prediction_candidates[0] == '{zero} l1 {defi} l4 {indef} l5'.format(
        zero=Constants.no_article_token, indef=Constants.article_indefinite_bucket_token, defi=Constants.article_definite_token
    )
    assert prediction_candidates[1] =='{zero} l1 {defi} l4 {defi} l5'.format(
        zero=Constants.no_article_token, defi=Constants.article_definite_token
    )
    assert prediction_candidates[2] == '{zero} l1 {defi} l4 {zero} l5'.format(
        zero=Constants.no_article_token, defi=Constants.article_definite_token
    )


def test_sentence_state_tracker_zero_filled2():
    t = Tree.from_string('(TOP~is~1~1 (S~is~2~2 (NPB~Vinken~2~2 (NNP Mr.) (NNP Vinken)) (VP~is~2~1 (VBZ is) (NP~chairman~2~1 (NPB~chairman~1~1 (NN chairman)) (PP~of~2~1 (IN of) (NP~N.V.~2~1 (NPB~N.V.~2~2 (NNP Elsevier) (NNP N.V.) (, ,)) (NPB~group~4~4 (DT the) (JJ Dutch) (NN publishing) (NN group) (. .))))))))')
    tokens = [Constants.no_article_token] + ' {} '.format(Constants.no_article_token).join(t.get_words()).split(' ')
    stt = SentenceStateTracker(tokens, zero_filled=True)

    bnp = next((n for n in t if n.get_label() == 'NPB'))
    prediction_candidates = stt.get_prediction_candidates(bnp)
    assert prediction_candidates[0].startswith('{indef} Mr. {zero} Vinken {zero} is {zero} chairman'.format(
        zero=Constants.no_article_token, indef=Constants.article_indefinite_bucket_token
    ))
    assert prediction_candidates[1].startswith('{defi} Mr. {zero} Vinken {zero} is {zero} chairman'.format(
        zero=Constants.no_article_token, defi=Constants.article_definite_token
    ))
    assert prediction_candidates[2].startswith('{zero} Mr. {zero} Vinken {zero} is {zero} chairman'.format(
        zero=Constants.no_article_token
    ))
    stt.record_prediction(bnp, Constants.article_indefinite_bucket_token)
    assert stt.last_prediction_idx == 1
    assert stt.so_far_predicted == [Constants.article_indefinite_bucket_token]

    bnp = next((n for n in t if n.val == 'NPB~chairman~1~1'))
    prediction_candidates = stt.get_prediction_candidates(bnp)
    assert prediction_candidates[0].startswith('{indef} Mr. {zero} Vinken {zero} is {indef} chairman {zero} of'.format(
        zero=Constants.no_article_token, indef=Constants.article_indefinite_bucket_token
    ))
    assert prediction_candidates[1].startswith('{indef} Mr. {zero} Vinken {zero} is {defi} chairman'.format(
        zero=Constants.no_article_token, defi=Constants.article_definite_token, indef=Constants.article_indefinite_bucket_token
    ))
    assert prediction_candidates[2].startswith('{indef} Mr. {zero} Vinken {zero} is {zero} chairman'.format(
        zero=Constants.no_article_token, indef=Constants.article_indefinite_bucket_token
    ))
    stt.record_prediction(bnp, Constants.article_definite_token)
    assert stt.last_prediction_idx == 7
    assert stt.so_far_predicted == [
        Constants.article_indefinite_bucket_token,
        'Mr.',
        Constants.no_article_token,
        'Vinken',
        Constants.no_article_token,
        'is',
        Constants.article_definite_token
    ]


def test_sentence_state_tracker_no_filled():
    t = Tree.from_string('(R (L1 l1) (L2 (L3 (L4 l4))) (L5 l5))')
    tokens = t.get_words()
    stt = SentenceStateTracker(tokens, zero_filled=False)

    bnp = next((n for n in t if n.get_label() == 'L2'))
    prediction_candidates = stt.get_prediction_candidates(bnp)
    assert prediction_candidates[0] == 'l1 {indef} l4 l5'.format(
        zero=Constants.no_article_token, indef=Constants.article_indefinite_bucket_token
    )
    assert prediction_candidates[1] =='l1 {defi} l4 l5'.format(
        zero=Constants.no_article_token, defi=Constants.article_definite_token
    )
    assert prediction_candidates[2] == 'l1 l4 l5'.format(
        zero=Constants.no_article_token
    )

    stt.record_prediction(bnp, Constants.article_definite_token)
    assert stt.last_prediction_idx == 1
    assert stt.so_far_predicted == ['l1', Constants.article_definite_token]

    bnp = next((n for n in t if n.get_label() == 'L5'))
    prediction_candidates = stt.get_prediction_candidates(bnp)
    assert prediction_candidates[0] == 'l1 {defi} l4 {indef} l5'.format(
        zero=Constants.no_article_token, indef=Constants.article_indefinite_bucket_token, defi=Constants.article_definite_token
    )
    assert prediction_candidates[1] =='l1 {defi} l4 {defi} l5'.format(
        zero=Constants.no_article_token, defi=Constants.article_definite_token
    )
    assert prediction_candidates[2] == 'l1 {defi} l4 l5'.format(
        zero=Constants.no_article_token, defi=Constants.article_definite_token
    )

    stt.record_prediction(bnp, Constants.article_definite_token)
    assert stt.last_prediction_idx == 2
    assert stt.so_far_predicted == ['l1', Constants.article_definite_token, 'l4', Constants.article_definite_token]


def test_sentence_state_tracker_no_filled2():
    t = Tree.from_string('(TOP~is~1~1 (S~is~2~2 (NPB~Vinken~2~2 (NNP Mr.) (NNP Vinken)) (VP~is~2~1 (VBZ is) (NP~chairman~2~1 (NPB~chairman~1~1 (NN chairman)) (PP~of~2~1 (IN of) (NP~N.V.~2~1 (NPB~N.V.~2~2 (NNP Elsevier) (NNP N.V.) (, ,)) (NPB~group~4~4 (DT the) (JJ Dutch) (NN publishing) (NN group) (. .))))))))')
    tokens = t.get_words()
    stt = SentenceStateTracker(tokens, zero_filled=False)

    bnp = next((n for n in t if n.get_label() == 'NPB'))
    prediction_candidates = stt.get_prediction_candidates(bnp)
    assert prediction_candidates[0].startswith('{indef} Mr. Vinken is chairman'.format(
        zero=Constants.no_article_token, indef=Constants.article_indefinite_bucket_token
    ))
    assert prediction_candidates[1].startswith('{defi} Mr. Vinken is chairman'.format(
        zero=Constants.no_article_token, defi=Constants.article_definite_token
    ))
    assert prediction_candidates[2].startswith('Mr. Vinken is chairman'.format(
        zero=Constants.no_article_token
    ))
    stt.record_prediction(bnp, Constants.article_indefinite_bucket_token)
    assert stt.last_prediction_idx == 0
    assert stt.so_far_predicted == [Constants.article_indefinite_bucket_token]

    bnp = next((n for n in t if n.val == 'NPB~chairman~1~1'))
    prediction_candidates = stt.get_prediction_candidates(bnp)
    assert prediction_candidates[0].startswith('{indef} Mr. Vinken is {indef} chairman of'.format(
        zero=Constants.no_article_token, indef=Constants.article_indefinite_bucket_token
    ))
    assert prediction_candidates[1].startswith('{indef} Mr. Vinken is {defi} chairman'.format(
        zero=Constants.no_article_token, defi=Constants.article_definite_token, indef=Constants.article_indefinite_bucket_token
    ))
    assert prediction_candidates[2].startswith('{indef} Mr. Vinken is chairman'.format(
        zero=Constants.no_article_token, indef=Constants.article_indefinite_bucket_token
    ))
    stt.record_prediction(bnp, None)
    assert stt.last_prediction_idx == 3
    assert stt.so_far_predicted == [Constants.article_indefinite_bucket_token, 'Mr.', 'Vinken', 'is']

    bnp = next((n for n in t if n.val == 'NPB~N.V.~2~2'))
    prediction_candidates = stt.get_prediction_candidates(bnp)
    assert prediction_candidates[0].startswith('{indef} Mr. Vinken is chairman of {indef} Elsevier'.format(
        zero=Constants.no_article_token, indef=Constants.article_indefinite_bucket_token
    ))
    assert prediction_candidates[1].startswith('{indef} Mr. Vinken is chairman of {defi} Elsevier'.format(
        zero=Constants.no_article_token, defi=Constants.article_definite_token, indef=Constants.article_indefinite_bucket_token
    ))
    assert prediction_candidates[2].startswith('{indef} Mr. Vinken is chairman of Elsevier'.format(
        zero=Constants.no_article_token, indef=Constants.article_indefinite_bucket_token
    ))
    stt.record_prediction(bnp, Constants.article_definite_token)
    assert stt.last_prediction_idx == 5
    assert stt.so_far_predicted == [Constants.article_indefinite_bucket_token, 'Mr.', 'Vinken', 'is', 'chairman', 'of', 'the']
