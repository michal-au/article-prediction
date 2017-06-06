import sys
import os
import codecs
from code.lib.language_model import kenlm
from code.lib.utils import read_settings
from code.corrector.prepare_data import get_sentences, remove_articles
from code.lib.kenlm_sentence_predictor import predict_sentence
from code.lib.eval_predicted_text import eval_predicted_text
import nltk
import argparse


SETTINGS = read_settings()

print("LOADING KENLM....")
KENLM_MODEL_PATH = os.path.join(SETTINGS.get('paths', 'modelLM'), 'kenlm-ggl-5-nbs-cls3')
model = kenlm.Model(KENLM_MODEL_PATH)
print("...DONE")

PENN_DATA_PATH = os.path.join(SETTINGS.get('paths', 'dataCorrector'), 'penn')
LOG_RESULTS_PATH = os.path.join(SETTINGS.get('paths', 'logModelResults'), 'penn', 'corrector')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--threshold')
    parser.add_argument('--margin')
    args = parser.parse_args()
    threshold = args.threshold
    margin = args.margin

    model = kenlm.Model(KENLM_MODEL_PATH)
    sents = remove_articles(get_sentences(os.path.join(PENN_DATA_PATH, 'penn_test_raw_orig.txt'), sent_tokenize=False))
    with open(os.path.join(LOG_RESULTS_PATH, 'test_lm.csv'), 'a+') as f:
        f.write('threshold, margin, errors, corrects, swaps, insertions, deletions\n')
        print("Treshold: {}, margin: {}".format(threshold, margin))
        swaps, insertions, deletions, corrects = 0, 0, 0, 0
        for sent in sents:
            tokenized_orig_sent = nltk.word_tokenize(sent)
            pred_sent = predict_sentence(model, tokenized_orig_sent)
            s, i, d, c, sent_with_errors = eval_predicted_text(sent, pred_sent, tokenize=False)
            swaps += s
            insertions += i
            deletions += d
            corrects += c
        errors = swaps + insertions + deletions
        f.write(', '.join([str(i) for i in [threshold, margin, errors, corrects, swaps, insertions, deletions]]) + '\n')

        threshold, margin = 50, 0.2
        print("Treshold: {}, margin: {}".format(threshold, margin))
        swaps, insertions, deletions, corrects = 0, 0, 0, 0
        for sent in sents:
            tokenized_orig_sent = nltk.word_tokenize(sent)
            pred_sent = predict_sentence(model, tokenized_orig_sent)
            s, i, d, c, sent_with_errors = eval_predicted_text(sent, pred_sent, tokenize=False)
            swaps += s
            insertions += i
            deletions += d
            corrects += c
        errors = swaps + insertions + deletions
        f.write(', '.join([str(i) for i in [threshold, margin, errors, corrects, swaps, insertions, deletions]]) + '\n')

        threshold, margin = 50, 0.5
        print("Treshold: {}, margin: {}".format(threshold, margin))
        swaps, insertions, deletions, corrects = 0, 0, 0, 0
        for sent in sents:
            tokenized_orig_sent = nltk.word_tokenize(sent)
            pred_sent = predict_sentence(model, tokenized_orig_sent)
            s, i, d, c, sent_with_errors = eval_predicted_text(sent, pred_sent, tokenize=False)
            swaps += s
            insertions += i
            deletions += d
            corrects += c
        errors = swaps + insertions + deletions
        f.write(', '.join([str(i) for i in [threshold, margin, errors, corrects, swaps, insertions, deletions]]) + '\n')
