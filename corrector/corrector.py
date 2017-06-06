from prepare_data import get_sentences, normalize_special_characters, parse_sents, remove_articles

from collections import deque
import nltk
import os
import pandas
import scipy
import sys
import cPickle
from nltk.parse.stanford import StanfordParser

from ..lib import utils
from ..lib.Tree import Tree
# from ..extract_features.extract_bnc import np_selector
# from ..extract_features.extract import populate_feature_dict, _add_to_history
# from ..experiments.bnc.tune import features as FEATURES
# from ..experiments.df_transform.embeddings import process_embeddings
# from ..extract_features.list_to_binary_features import prepare_for_vectorizer, convert_bools





# SETTINGS = utils.read_settings()
#
# # model = cPickle.load(open(os.path.join(SETTINGS.get('paths', 'model'), 'model.pkl'), 'r'))
# # vectorizer = cPickle.load(open(os.path.join(SETTINGS.get('paths', 'model'), 'vectorizer.pkl'), 'r'))
#
#
# def remove_articles(sentence):
#     if len(sentence) <= 1:
#         return sentence
#     return ' '.join([w for w in sentence.split() if w.lower() not in ('a', 'an', 'the')])

#
# def extract_features(trees):
#     feature_collection = []
#     history = deque([None] * 5)
#
#     for t_num, t in enumerate(trees, start=1):
#         context_words = t.get_word_tag_pairs()
#         for n_num, node in enumerate(t, start=1):
#             if np_selector(node):
#                 example_features = populate_feature_dict(node, history, False, context_words, article_length=len(trees), node_nb=t_num - 1)
#                 # example_features['_coordinates'] = f_path + "|" + str(l_num) + "|" + str(n_num)
#                 example_features['_sent'] = node.print_highlighted()
#                 feature_collection.append(example_features)
#
#             history = _add_to_history(history, t)
#
#     df = pandas.DataFrame(feature_collection)
#
#     features_to_drop = [f for f in df.columns if f not in FEATURES]
#     df = df.drop(features_to_drop, 1)
#     df, _ = process_embeddings(df, FEATURES)
#     df = convert_bools(df)
#     df, dict_data = prepare_for_vectorizer(df)
#     vec_array = vectorizer.transform(dict_data)
#     return scipy.sparse.hstack([scipy.sparse.csr_matrix(df.values), vec_array], format='csr')
#
#
# def format_output(y, trees):
#     idx = 0
#     for t in trees:
#         nodes = []
#         for n in t:
#             if np_selector(n):
#                 nodes.append((n, y[idx]))
#                 idx += 1
#         for n, art in nodes:
#             if art == 'ZERO':
#                 continue
#             if art == 'A/AN':
#                 n.insert_article('a')
#                 continue
#             n.insert_article('the')
#
#         print t.print_sentence()
#
#     assert idx == len(y)
#
#
# def correct_file(f_name):
#     f_in = codecs.open(f_name, 'r', 'utf-8')
#     sents = []
#     for l in f_in:
#         sents.extend(tokenizer.tokenize(l.strip()))
#
#     print "... removing articles"
#     sents = [remove_articles(s) for s in sents if len(s.split()) <= MAXLENGTH]
#
#     print "... parsing"
#     trees = parser.raw_parse_sents(sents[:200])
#     trees = [Tree.from_string(str(t_)) for t in trees for t_ in t]
#     print trees
#     #
#     # print "...extracting features"
#     # df = extract_features(trees)
#     #
#     # print "...predicting"
#     # y = model.predict(df)
#     #
#     # format_output(y, trees)
#

if __name__ == '__main__':
    sents = get_sentences(sys.argv[1], sent_tokenize=True)

    #sents = normalize_special_characters(sents)
    # naparsujeme original, abychom nemuseli resit blbosti typu 'LLR'?: trees = parse_sents(sents)
    #sents = remove_articles(sents)
    #print(sents[:200])
    #trees = parse_sents(sents)
    #for t in trees:
    #    print(t.to_string())
