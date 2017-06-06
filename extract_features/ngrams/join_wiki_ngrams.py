import os
import codecs
import cPickle

from .create_wiki_ngrams import init_ngrams
from ...lib.utils import read_settings

STEP = 90
SETTINGS = read_settings()
WIKI_PARTIAL_NGRAMS_PATH = SETTINGS.get('paths', 'dataNgramsPartial')


def join_ngrams(final_ngrams, ngrams):
    for k in ngrams:
        for kk in ngrams[k]:
            for kkk in ngrams[k][kk]:
                for art in ngrams[k][kk][kkk]:
                    final_ngrams[k][kk][kkk][art] += ngrams[k][kk][kkk][art]
    return final_ngrams


if __name__ == '__main__':
    ngrams_final = init_ngrams()
    f_paths = [os.path.join(WIKI_PARTIAL_NGRAMS_PATH, f_name) for f_name in os.listdir(WIKI_PARTIAL_NGRAMS_PATH)]
    files_count = len(f_paths)
    i = 1
    for idx, f_path in enumerate(f_paths, start=1):
        print f_path, idx/float(files_count)
        ngrams_partial = cPickle.load(open(f_path, 'r'))
        ngrams_final = join_ngrams(ngrams_final, ngrams_partial)
        if idx % STEP == 0:
            cPickle.dump(ngrams_final, open(os.path.join(
                SETTINGS.get('paths', 'dataNgrams'), 'tmp-wiki-counts{}.pkl'.format(i)), 'wb'
            ))
            i += 1
            ngrams_final = init_ngrams()

    cPickle.dump(ngrams_final, open(os.path.join(
        SETTINGS.get('paths', 'dataNgrams'), 'tmp-wiki-counts{}.pkl'.format(i)), 'wb'
    ))
