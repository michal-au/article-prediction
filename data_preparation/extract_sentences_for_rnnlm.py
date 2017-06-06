import os

from ..lib import Tree
from ..lib import corpus
from ..lib import utils
from ..lib.Constants import Constants


def sentences_from_trees(trees):
    sents = []
    for t in trees:
        sent = []
        for token in t.removeNodeByValue('-NONE-').get_words():
            token = token.lower()
            try:
                float(token)
                token = "<number>"
            except ValueError:
                pass

            if (
                (token in Constants.article_tokens) or
                (sent and sent[-1] in (Constants.article_indefinite_bucket_token, Constants.article_definite_token))
            ):
                if token in Constants.article_indefinite_tokens:
                    # 'a', 'an' -> 'a/an'
                    token = Constants.article_indefinite_bucket_token
                sent.append(token)
            else:
                sent.extend([Constants.no_article_token, token])
        sents.append(' '.join(sent))

    return sents


def extract_sentences(old_path):
    """extract tokenized sentences from Penn Treebank 3 wsj/**/*.mrg files"""
    trees = Tree.Tree.from_file(old_path)
    return sentences_from_trees(trees)


if __name__ == '__main__':
    settings = utils.read_settings()
    orig_path = settings.get('paths', 'dataOrig')
    new_path = settings.get('paths', 'dataRnnlmRaw')
    sentences = []
    for r, ds, fs in os.walk(orig_path):
        print r
        if r.endswith(Constants.penn_leftout_dir):
            continue
        ds.sort()
        fs.sort()
        for f in fs:
            orig_file = os.path.join(r, f)
            sentences.extend(extract_sentences(orig_file))
        if r.endswith(Constants.penn_last_train_dir):
            utils.print_list_to_file(sentences, os.path.join(new_path, 'train'))
            sentences = []
        elif r.endswith(Constants.penn_heldout_dir):
            utils.print_list_to_file(sentences, os.path.join(new_path, 'heldout'))
            sentences = []
        elif r.endswith(Constants.penn_test_dir):
            utils.print_list_to_file(sentences, os.path.join(new_path, 'test'))
            sentences = []
