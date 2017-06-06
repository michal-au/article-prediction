import nltk
from prepare_data import get_sentences, normalize_special_characters, parse_sents, remove_articles
import os
from code.lib import utils
import argparse
from code.lib.eval_predicted_text import eval_predicted_text
import codecs
from code.lib.Tree import Tree
#from extract_features import np_selector
from collections import defaultdict


SETTINGS = utils.read_settings()

MANUAL_DATA_PATH = os.path.join(SETTINGS.get('paths', 'dataCorrector'), 'manual')
LOG_RESULTS_PATH = os.path.join(SETTINGS.get('paths', 'logModelResults'), 'bnc_manual_comparison')
TOTAL_ART_POSITIONS = 163


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

    #print u' '.join([w if isinstance(w, str) else unicode(w) for w in error_repr])
    return u' '.join([w if isinstance(w, str) else unicode(w) for w in error_repr])

def print_confmat(cm):
    # for true in cm:
    #     print(',' + ', '.join(pred for pred in cm[true]))
    #     break
    categories = ('the', 'a/an', 'ZERO')
    print(',' + ', '.join(categories))
    for true in categories:
        print(true + ', ' + ', '.join(str(cm[true][pred]/float(1))for pred in categories))
    #for true in cm:
    #    print(true + ', ' + ', '.join(pred + ":" + str(cm[true][pred]) for pred in cm[true]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess text before it is used for language model')
    parser.add_argument('--parse-fillin-orig-noarticles', action='store_true')
    parser.add_argument('--show-automatic-parse-candidates-for-prediction', action='store_true')
    parser.add_argument('--all-together', action='store_true')
    args = parser.parse_args()

    orig_sents = get_sentences(os.path.join(MANUAL_DATA_PATH, 'fill_in_orig.txt'), sent_tokenize=True)

    if args.parse_fillin_orig_noarticles:
        orig_sents = normalize_special_characters(orig_sents)
        trees = parse_sents(orig_sents)
        with open(os.path.join(MANUAL_DATA_PATH, 'fill_in_orig_parsed.txt'), 'w+') as f:
            for t in trees:
                f.write(t.to_string() + '\n')
        exit()

    if args.show_automatic_parse_candidates_for_prediction:
        with codecs.open(os.path.join('code', 'corrector', 'tmp_parses.txt'), 'r', 'utf-8') as f:
            print [t for t in f.readlines()]
            trees = [Tree.from_string(t) for t in f.readlines()]
            #for t in trees:
            #    print [n for n in t if n.is_leaf()]
        # for t in trees:
        #     for n in t:
        #         if np_selector(n):
        #             print n.print_highlighted()
        exit()

    if args.all_together:
        predict_sents = [
            get_sentences(os.path.join(MANUAL_DATA_PATH, 'fill_in',  predict_file), sent_tokenize=True)
            for predict_file in ('solution_D.txt',)
            #for predict_file in ('solution_A.txt', 'solution_B.txt', 'solution_C.txt', 'solution_D.txt')
            #for predict_file in ('solution_LOGREG.txt', 'solution_XGB.txt')
        ]
        for annotator in predict_sents:
            assert len(orig_sents) == len(annotator)

        new_sents = []
        for sent_nb in range(len(orig_sents)):
            new_sents.append(compare_multiple_predictions(orig_sents[sent_nb], [annotator[sent_nb].replace(u'\u2019', "'") for annotator in predict_sents]))
        with codecs.open(os.path.join('code', 'corrector', 'tmp_errors_in_manual_annot.txt'), 'w', 'utf-8') as f:
            for s in new_sents:
                f.write(s + u'\n')
        exit()

    with open(os.path.join(LOG_RESULTS_PATH, 'fillin_all_models.csv'), 'a+') as f:

        cm = defaultdict(lambda: defaultdict(int))
        for predict_file in (
                'solution_XGB.txt',
                #'solution_A.txt', 'solution_B.txt', 'solution_C.txt', 'solution_D.txt',
                #'solution_A.txt', 'solution_B.txt', 'solution_C.txt', 'solution_D.txt', 'solution_LOGREG.txt', 'solution_XGB.txt', 'solution_BASELINE.txt',
                #'solution_LM.txt', 'solution_LM50.txt', 'solution_LM100.txt', 'solution_LM200.txt', 'solution_LM300.txt', 'solution_LM350.txt', 'solution_LM400.txt', 'solution_LM450.txt', 'solution_LM500.txt', 'solution_LM600.txt',
        ):
            predict_sents = get_sentences(os.path.join(MANUAL_DATA_PATH, 'fill_in',  predict_file), sent_tokenize=True)
            assert len(orig_sents) == len(predict_sents)

            swaps, insertions, deletions, corrects = 0, 0, 0, 0
            for orig_sent, predict_sent in zip(orig_sents, predict_sents):
                predict_sent = predict_sent.replace(u'\u2019', "'")
                s, i, d, c, sent_with_errors, cm = eval_predicted_text(orig_sent, predict_sent, cv=cm)
                swaps += s
                insertions += i
                deletions += d
                corrects += c
                sent_with_errors = ' '.join(sent_with_errors)

            errors = swaps + insertions + deletions
            print errors, corrects
            #f.write(', '.join([str(i) for i in [predict_file, float(TOTAL_ART_POSITIONS-errors)/TOTAL_ART_POSITIONS, corrects, errors, swaps, insertions, deletions]]) + '\n')

    print_confmat(cm)