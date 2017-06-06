import sys
import os
import codecs
from code.lib.language_model import kenlm
from code.lib.utils import read_settings
from code.corrector.prepare_data import get_sentences, remove_articles
from code.lib.kenlm_sentence_predictor import predict_sentence
from code.lib.eval_predicted_text import eval_predicted_text
import nltk


PRPLX_THRESHOLD_VALUES = (0, 50, 100, 150, 200)
PRPLX_MARGIN_RATIOS = (0, 0.05, 0.1, 0.2)

SETTINGS = read_settings()

print("LOADING KENLM....")
KENLM_MODEL_PATH = os.path.join(SETTINGS.get('paths', 'modelLM'), 'kenlm-ggl-5-nbs-cls3')
model = kenlm.Model(KENLM_MODEL_PATH)
print("...DONE")


if __name__ == "__main__":
    model = kenlm.Model(KENLM_MODEL_PATH)
    sents = remove_articles(get_sentences(sys.argv[1]))

    log_file_path = os.path.join(
        SETTINGS.get('paths', 'logKennlmResults'), 'penn_tuning_{}x{}.csv'.format('-'.join([str(t) for t in PRPLX_THRESHOLD_VALUES]), '-'.join([str(m) for m in PRPLX_MARGIN_RATIOS]))
    )
    with open(log_file_path, 'a') as f:
        f.write('threshold, margin, corrects, errors, swaps, insertions, deletions\n')
        for threshold in PRPLX_THRESHOLD_VALUES:
            for margin in PRPLX_MARGIN_RATIOS:
                print("Threshold: {}, margin: {}".format(threshold, margin))
                swaps, insertions, deletions, corrects = 0, 0, 0, 0
                for sent in sents:
                    tokenized_orig_sent = nltk.word_tokenize(sent)
                    pred_sent = predict_sentence(model, tokenized_orig_sent, threshold=threshold, margin=margin)
                    #print tokenized_orig_sent,
                    #print pred_sent
                    s, i, d, c, sent_with_errors = eval_predicted_text(tokenized_orig_sent, pred_sent, tokenize=False)
                    swaps += s
                    insertions += i
                    deletions += d
                    corrects += c
                errors = swaps + insertions + deletions
                f.write(', '.join([str(i) for i in [threshold, margin, corrects, errors, swaps, insertions, deletions]]) + '\n')
