"""
It seems this file is no longer needed, it was meant to download google ngrams stats, but they are too big.
"""

from collections import defaultdict
import os
import urllib

from ...lib.utils import read_settings, read_trees_from_file

BASE_URL = 'http://storage.googleapis.com/books/ngrams/books/'
FILE_URLNAME_TEMPLATE = 'googlebooks-eng-all-{}gram-20120701-{}.gz'

SETTINGS = read_settings()
LOCAL_FILES_PATH = SETTINGS.get('paths', 'dataNgramsRawFiles')

NP_SELECTOR = lambda x: x.get_label() == 'NPB'


def get_ngrams_for_np(np, i, length=3):
    leftmost_child = np.get_leftmost_child()
    idx = leftmost_child.order_nb
    root = np.get_root()
    leaves = root.get_words()
    start_idx = max(0, idx - i)

    if leftmost_child.get_word_form().lower() in ('the', 'a', 'an'):
        n_gram = leaves[start_idx: start_idx + length + 1]
        n_gram.remove(leftmost_child.get_word_form().lower())
    else:
        n_gram = leaves[start_idx: start_idx + length]
    print i, n_gram
    return n_gram


def get_ngram_starting_chars_from_parsed_data(f_paths):
    chars_pairs = defaultdict(int)
    for f_path in f_paths:
        trees = read_trees_from_file(f_path)
        for t in trees:
            for n in t:
                if NP_SELECTOR(n):
                    n_gram_a = get_ngrams_for_np(n, 2)
                    n_gram_b = get_ngrams_for_np(n, 1)
                    if len(n_gram_a) == 3:
                        chars_pairs[n_gram_a[0][:2]] += 1
                    if len(n_gram_b) == 3:
                        chars_pairs[n_gram_b[0][:2]] += 1

    return chars_pairs


def download_ngram_files(starting_chars):
    files_on_disk = {os.path.splitext(fname)[0] for fname in os.listdir(LOCAL_FILES_PATH)}
    files_to_download = set(starting_chars) - files_on_disk
    for f in files_to_download:
        print f
        for n_gram in (4, 5):
            f_name = FILE_URLNAME_TEMPLATE.format(n_gram, f)
            urllib.urlretrieve(
                os.path.join(BASE_URL, f_name),
                os.path.join(LOCAL_FILES_PATH, f_name)
            )


if __name__ == '__main__':
    #download_ngram_files(['as', 'to'])
    path = SETTINGS.get('paths', 'dataParsed')
    f_paths = []  # the resulting feature table
    for r, ds, fs in os.walk(path):
        if r in [os.path.join(path, '24')]:
            continue
        ds.sort()
        fs.sort()
        for f in fs:
            f_path = os.path.join(r, f)
            f_paths.append(f_path)

    chrs = get_ngram_starting_chars_from_parsed_data(f_paths[:2])
    for ch in sorted(chrs, key=chrs.get, reverse=True):
        print ch, chrs[ch]
