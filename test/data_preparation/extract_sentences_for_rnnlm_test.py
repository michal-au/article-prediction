import os

from ...lib.utils import read_settings
from ...data_preparation.extract_sentences_for_rnnlm import sentences_from_trees
from ...lib.Tree import Tree
from ...lib.Constants import Constants

def test_sentences_from_trees():
    trees = [
        Tree.from_string('(BNP (DT the) (ADJ perfect) (NN example) )'),
        Tree.from_string('(BNP (DT a) (ADJ A) (NN a) )'),
        Tree.from_string('(BNP (DT huh))'),
        Tree.from_string('(BNP (DT cool) (DT sentence) (DT with) (DT a) (DT number) (DT 4333))'),
    ]
    assert sentences_from_trees(trees) == [
        'the perfect {} example'.format(Constants.no_article_token),
        'a a a',
        '{} huh'.format(Constants.no_article_token),
        '{zero} cool {zero} sentence {zero} with a number {zero} <number>'.format(zero=Constants.no_article_token)
    ]
