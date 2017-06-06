# -*- coding: utf-8 -*-
import nltk
from nltk.parse.stanford import StanfordParser
import codecs
import os.path

from code.lib import utils
from code.data_preparation.parse.to_ascii import to_ascii
from ..lib.Tree import Tree


SETTINGS = utils.read_settings()
MAXLENGTH = 200  # number of words in the longest possible sentence. (longer sentences will be discarded)

stanford_parser_dir = os.path.join(os.getcwd(), SETTINGS.get('paths', 'stanfordParser'))
my_path_to_jar = os.path.join(stanford_parser_dir, 'stanford-parser.jar')
my_path_to_models_jar = os.path.join(stanford_parser_dir, 'stanford-parser-3.6.0-models.jar')
eng_model_path = os.path.join(stanford_parser_dir, 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

PARSER = StanfordParser(
    model_path=eng_model_path,
    path_to_models_jar=my_path_to_models_jar,
    path_to_jar=my_path_to_jar,
    java_options='-mx5000m'
)
PARSER._classpath = tuple([j for j in PARSER._classpath] + [stanford_parser_dir + '/slf4j-api.jar', stanford_parser_dir + '/slf4j-simple.jar'])
SENT_TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle')


def get_sentences(f_path, sent_tokenize=False):
    with codecs.open(f_path, 'r', 'utf-8') as f:
        sents = [sent.strip() for sent in f.readlines()]
    if sent_tokenize:
        sents = SENT_TOKENIZER.tokenize(' '.join(sents))
    return sents


def normalize_special_characters(sents):
    new_sents = []
    for sent in sents:
        new_sents.append(''.join([to_ascii.get(ch.encode('utf-8'), ch) for ch in sent]))
    return new_sents


def remove_articles(sents):
    new_sents = []
    for sent in sents:
        if len(sent) <= 1:  # hack, nechceme odstranit radek na kterym ja A
            new_sents.append(sent)
        new_sents.append(' '.join([w for w in sent.split() if w.lower() not in ('a', 'an', 'the')]))
    return new_sents


def parse_sents(sents):
    trees = PARSER.raw_parse_sents(sents)
    trees = [Tree.from_string(str(t_)) for t in trees for t_ in t]
    return trees
