from ..lib import Tree
from ..lib import corpus
from ..lib import utils


def extract_sentences(old_path, new_path):
    """extract tokenized sentences from Penn Treebank 3 wsj/**/*.mrg files"""
    trees = Tree.Tree.from_file(old_path)
    sents = [' '.join(t.removeNodeByValue('-NONE-').get_words()) for t in trees]
    utils.print_list_to_file(sents, new_path)


if __name__ == '__main__':
    settings = utils.read_settings()
    old_path = settings.get('paths', 'dataOrig')
    new_path = settings.get('paths', 'dataRaw')

    corpus.walk_and_transform(extract_sentences, old_path, new_path)
