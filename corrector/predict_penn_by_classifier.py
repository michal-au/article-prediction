# -*- coding: utf-8 -*-
import codecs
from prepare_data import get_sentences, normalize_special_characters, parse_sents, remove_articles
import os
from code.lib import utils
import argparse
import pickle
from code.lib.kenlm_sentence_predictor import predict_sentence
import nltk
import numpy as np
from code.lib.eval_predicted_text import eval_predicted_text
from xgboost.sklearn import XGBClassifier
from code.lib.language_model import kenlm
from extract_features import extract_features, np_selector
#TODO: smazat az nebudu nacitat stromy z tmp filu
from code.lib.Tree import Tree


SETTINGS = utils.read_settings()
XGB_MODEL_PATH = os.path.join(SETTINGS.get('paths', 'model'), 'xgboost_final_263088.pkl')
KENLM_MODEL_PATH = os.path.join(SETTINGS.get('paths', 'modelLM'), 'kenlm-ggl-5-nbs-cls3')


def print_sentence(tree):
    s = u''
    for idx, n in enumerate(tree):
        if not n.is_leaf():
            continue
        token = n.get_word_form()
        if token == '-LRB-':
            token = u'('
        if token == '-RRB-':
            token = u')'
        if token == '#':
            token = u'£'
        if token == '--':
            token = u'—'
        if n.get_label() in (',', '.', '!', '?', ';', ':', "'s", '-RRB-',):
            if not token == u'—':
                s += unicode(token)
            else:
                s += u' ' + unicode(token)
        elif token[0].isdigit() and s and s[-1] == u'£':
            s += unicode(token)
        elif s and s[-1] == u'(':
            s += unicode(token)
        else:
            s += u' ' + unicode(token)
    return s


def format_predictions_by_machlearn_model(y, trees):
    new_sents = []
    idx = 0
    for t in trees:
        nodes = []
        for n in t:
            if np_selector(n):
                nodes.append((n, y[idx]))
                idx += 1
        for n, art in nodes:
            if art == 'ZERO':
                continue
            if art == 'A/AN':
                n.insert_article('a')
                continue
            n.insert_article('the')
        new_sents.append(print_sentence(t))
    assert idx == len(y)
    return new_sents


def predict_by_logreg(x):
    target_tuple = ('THE', 'A', 'ZERO')
    print type(x), x.shape
    predictions = np.zeros((x.shape[0], 3))
    for i, target in enumerate(target_tuple):
        m = pickle.load(open(os.path.join(
            SETTINGS.get('paths', 'model'),
            'logreg_ovr_binarized_{}_allfeatures_263088.pkl'.format(target)
        ), 'rb'))
        assert m.classes_[0] == target
        predictions[:, i] = m.predict_proba(x)[:, 0]
    return [target_tuple[idx] for idx in np.argmax(predictions, axis=1)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    sents = remove_articles(get_sentences(args.input, sent_tokenize=True))
    model = None
    if args.model_type == 'xgboost':
        sents = remove_articles(get_sentences(args.input, sent_tokenize=False))
        model = pickle.load(open(XGB_MODEL_PATH, 'rb'))
        trees = parse_sents(sents)
        assert len(trees) == len(sents)
        # #trees = []
        # #with codecs.open(os.path.join('code', 'corrector', 'tmp_parses.txt'), 'r', 'utf-8') as f:
        # #    trees = [Tree.from_string(t) for t in f.readlines()]
        test_x = extract_features(trees)
        test_y = model.predict(test_x)
        new_sents = format_predictions_by_machlearn_model(test_y, trees)
        with codecs.open(args.output, 'w+', 'utf-8') as f:
            for sent in new_sents:
                f.write(sent + '\n')
    if args.model_type == 'logreg':
        # trees = parse_sents(sents)
        #trees = []
        #with codecs.open(os.path.join('code', 'corrector', 'tmp_parses.txt'), 'r', 'utf-8') as f:
        #    trees = [Tree.from_string(t) for t in f.readlines()]
        sents = remove_articles(get_sentences(args.input, sent_tokenize=False))
        trees = parse_sents(sents)
        test_x = extract_features(trees)
        test_y = predict_by_logreg(test_x)
        print test_y
        new_sents = format_predictions_by_machlearn_model(test_y, trees)
        with codecs.open(args.output, 'w+', 'utf-8') as f:
            for sent in new_sents:
                f.write(sent + '\n')
    if args.model_type == 'kenlm':
        print("LOADING KENLM....")
        model = kenlm.Model(KENLM_MODEL_PATH)
        print("...DONE")
        with codecs.open(args.output, 'w+', 'utf-8') as f:
            new_sents = []
            for sent in sents:
                f.write(' '.join(predict_sentence(model, nltk.word_tokenize(sent), threshold=50)) + '\n')
        # print(predict_sentence(model, nltk.word_tokenize(sents[0])))

    # orig_sents = get_sentences(os.path.join(MANUAL_DATA_PATH, 'fill_in_orig.txt'), sent_tokenize=True)
    #
    # if args.parse_fillin_orig_noarticles:
    #     orig_sents = normalize_special_characters(orig_sents)
    #     trees = parse_sents(orig_sents)
    #     with open(os.path.join(MANUAL_DATA_PATH, 'fill_in_orig_parsed.txt'), 'w+') as f:
    #         for t in trees:
    #             f.write(t.to_string() + '\n')
    #     exit()
    #
    # for predict_file in ('solution_A.txt', 'solution_B.txt', 'solution_C.txt', 'solution_D.txt'):
    #     predict_sents = get_sentences(os.path.join(MANUAL_DATA_PATH, 'fill_in',  predict_file), sent_tokenize=True)
    #     assert len(orig_sents) == len(predict_sents)
    #
    #     swaps, insertions, deletions = 0, 0, 0
    #     for orig_sent, predict_sent in zip(orig_sents, predict_sents):
    #         s, i, d = eval_predicted_text(orig_sent, predict_sent)
    #         swaps += s
    #         insertions += i
    #         deletions += d
    #
    #     errors = swaps + insertions + deletions
    #     print float(TOTAL_ART_POSITIONS-errors)/TOTAL_ART_POSITIONS, errors, swaps, insertions, deletions
