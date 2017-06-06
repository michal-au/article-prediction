import nltk
from prepare_data import get_sentences#, normalize_special_characters, parse_sents, remove_articles
import os
from code.lib import utils
import argparse
from code.lib.eval_predicted_text import eval_predicted_text
import codecs
from code.lib.Tree import Tree
from extract_features import np_selector
import re


SETTINGS = utils.read_settings()

PENN_DATA_PATH = os.path.join(SETTINGS.get('paths', 'dataCorrector'), 'penn')
LOG_RESULTS_PATH = os.path.join(SETTINGS.get('paths', 'logModelResults'), 'penn', 'corrector')


def compare_multiple_predictions(orig_sent, pred_sents):
    """
    Vykresli smrsklou reprezentaci vsech predikci pro jednu originalni vetu: "sell-through market, as [the, .,.,a/an,X] retail sector is ..."
    Do znacne miry duplikace kodu z code.lib.eval_predicted_text :(
    """
    ARTICLE_TOKENS = ('a', 'an', 'a/an', 'the')
    error_repr = []
    orig_sent = nltk.word_tokenize(orig_sent)
    predict_sents = [nltk.word_tokenize(predict_sent) for predict_sent in pred_sents]
    predict_indices = [0] * len(predict_sents)
    orig_idx = 0
    while orig_idx < len(orig_sent) or any(predict_idx < len(predict_sents[idx]) for idx, predict_idx in enumerate(predict_indices)):
        orig_word = orig_sent[orig_idx].lower()
        predict_words = [predict_sent[predict_indices[idx]].lower() for idx, predict_sent in enumerate(predict_sents)]
        current_predictions = None
        if all(orig_word == predict_word for predict_word in predict_words):
            current_predictions = orig_word
            orig_idx += 1
            for idx in range(len(predict_indices)):
                predict_indices[idx] += 1
        elif orig_word in ARTICLE_TOKENS:
            current_predictions = [orig_word]
            for idx, predict_word in enumerate(predict_words):
                if predict_word == orig_word:
                    current_predictions.append('-')
                    predict_indices[idx] += 1
                elif predict_word in ARTICLE_TOKENS:
                    current_predictions.append(predict_word)
                    predict_indices[idx] += 1
                else:
                    current_predictions.append('X')
            orig_idx += 1
            current_predictions = current_predictions[0] + u'||' + u'|'.join(current_predictions[1:])
        else:
            current_predictions = ['-']
            for idx, predict_word in enumerate(predict_words):
                if predict_word == orig_word:
                    current_predictions.append('-')
                elif predict_word in ARTICLE_TOKENS:
                    current_predictions.append(predict_word)
                    predict_indices[idx] += 1
                else:
                    print(u"!!!: orig: >{}<, predict: >{}<".format(orig_word, predict_word))
                    print error_repr
                    raise ValueError(u"!!!: orig: {}, predict: {}".format(orig_word, predict_word))
            current_predictions = current_predictions[0] + u'||' + u'|'.join(current_predictions[1:])
        error_repr.append(current_predictions)

    print ' '.join([w if isinstance(w, str) else unicode(w) for w in error_repr])


def correct_stupid_formating(sent):
    # ad-hoc upravy kvuli blbymu formatovani pri predikci:
    sent = sent.replace('- owned', '-owned')
    sent = sent.replace('- a-year', '-a-year')
    sent = sent.replace('Co. . ', 'Co. ')
    sent = sent.replace('9:30 -10', '9:30-10')
    sent = sent.replace('1/2-inch', '1/2 - inch')

    return sent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--all-together', action='store_true')
    args = parser.parse_args()

    orig_sents = get_sentences(os.path.join(PENN_DATA_PATH, 'penn_test_raw_orig.txt'))

    if args.all_together:
        predict_sents = [
            get_sentences(os.path.join(PENN_DATA_PATH, predict_file))
            for predict_file in ('solution_LOGREG.txt', 'solution_XGB.txt', 'solution_LM_50_05.txt', 'solution_LM_0_0.txt')
        ]
        for annotator in predict_sents:
            assert len(orig_sents) == len(annotator)

        for sent_nb in range(len(orig_sents)):
            compare_multiple_predictions(orig_sents[sent_nb], [correct_stupid_formating(annotator[sent_nb]) for annotator in predict_sents])
        exit()

    with open(os.path.join(LOG_RESULTS_PATH, 'test.csv'), 'a+') as f:
        f.write('file, errors, corrects, swaps, insertions, deletions\n')
        for predict_file in (
                'solution_LOGREG.txt', 'solution_XGB.txt', 'solution_LM_50_05.txt', 'solution_LM_0_0.txt'
        ):
            predict_sents = get_sentences(os.path.join(PENN_DATA_PATH, predict_file))
            print(len(orig_sents), len(predict_sents))
            assert len(orig_sents) == len(predict_sents)

            swaps, insertions, deletions, corrects = 0, 0, 0, 0
            for orig_sent, predict_sent in zip(orig_sents, predict_sents):
                orig_sent = orig_sent.replace('1/2-inch', '1/2 - inch')
                predict_sent = correct_stupid_formating(predict_sent)

                s, i, d, c, sent_with_errors = eval_predicted_text(orig_sent, predict_sent)
                swaps += s
                insertions += i
                deletions += d
                corrects += c
                sent_with_errors = ' '.join(sent_with_errors)

            errors = swaps + insertions + deletions
            f.write(', '.join([str(i) for i in [predict_file, errors, corrects, swaps, insertions, deletions]]) + '\n')
