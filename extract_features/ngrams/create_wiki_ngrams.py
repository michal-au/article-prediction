from collections import defaultdict
import glob
import os
import codecs
import nltk
import cPickle
from functools import partial

from ...lib.utils import read_settings

BATCH_SIZE = 50000

SETTINGS = read_settings()
WIKI_TEXT_PATH = SETTINGS.get('paths', 'dataWikiRawText')


def process_word(w):
    w = w.lower()
    if w == "''":
        w = ''
    if w.startswith("''"):
        w = w[2:]
    if w.isdigit():
        w = '<number>'
    return w


def process_file(f_path, ngrams):
    with codecs.open(f_path, 'r', 'utf-8') as f:
        wds = nltk.word_tokenize(f.read())
        if not wds or len(wds) < 2:
            return ngrams
        words = [process_word(w) for w in wds if process_word(w)]
        for i in xrange(2, len(words) - 1):
            if words[i] in ('the', 'a', 'an'):
                ngrams[words[i - 2]][words[i - 1]][words[i + 1]][words[i]] += 1
            else:
                ngrams[words[i - 2]][words[i - 1]][words[i]]['ZERO'] += 1

    return ngrams


def print_ngrams(ngrams):
    for k in ngrams:
        for kk in ngrams[k]:
            for kkk in ngrams[k][kk]:
                for art in ngrams[k][kk][kkk]:
                    print k, kk, kkk, art, ngrams[k][kk][kkk][art]


def remove_tails(ngrams):
    for k in ngrams.keys():
        for kk in ngrams[k].keys():
            for kkk in ngrams[k][kk].keys():
                for art in ngrams[k][kk][kkk].keys():
                    if ngrams[k][kk][kkk][art] < 2:
                        del(ngrams[k][kk][kkk][art])
                if not len(ngrams[k][kk][kkk].keys()):
                    del(ngrams[k][kk][kkk])
            if not len(ngrams[k][kk].keys()):
                del(ngrams[k][kk])
        if not len(ngrams[k].keys()):
            del(ngrams[k])

    return ngrams


def init_ngrams():
    return defaultdict(partial(defaultdict, partial(defaultdict, partial(defaultdict, int))))


if __name__ == '__main__':
    ngrams = init_ngrams()
    i = 0
    for f_path in glob.iglob(os.path.join(WIKI_TEXT_PATH, '*')):
        i += 1
        #if i <= 8950000:
        #    continue
        if i % 100 == 0:
            print i

        ngrams = process_file(f_path, ngrams)

        if i % BATCH_SIZE == 0:
            ngrams = remove_tails(ngrams)
            print "SAVING ..."
            cPickle.dump(ngrams, open(os.path.join(
                SETTINGS.get('paths', 'dataNgrams'), 'wiki-counts-{}.pkl'.format(i)), 'wb'
            ))
            ngrams = init_ngrams()

    # last iteration only:
    ngrams = remove_tails(ngrams)
    print "SAVING ..."
    cPickle.dump(ngrams, open(os.path.join(
        SETTINGS.get('paths', 'dataNgrams'), 'wiki-counts-{}.pkl'.format(i)), 'wb'
    ))
