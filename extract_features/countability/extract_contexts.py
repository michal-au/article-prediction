from collections import defaultdict
import pickle
import os

from ...lib.wordnet import lemmatize
from ...lib.utils import read_settings
from ...lib.Tree import Tree


SETTINGS = read_settings()


class MODIFIERS:
    _DEMONSTRATIVES = {'these', 'those', 'this', 'that'}
    _POSSESSIVES = {'my', 'your', 'his', 'her', 'its', 'our', 'their'}
    _INTERROGATIVES = {'whose', 'what', 'which'}
    _QUANTIFIERS = {'more', 'most', 'no', 'some', 'all', 'any', 'enough', 'less'}

    COUNT = {'a', 'an', 'another', 'one', 'each'}
    MASS = {'much', 'less', 'enough', 'all', 'sufficient'}
    WHO_KNOWS = {'the'} | _DEMONSTRATIVES | _POSSESSIVES | _INTERROGATIVES | _QUANTIFIERS


def word_good_for_context(w, t):
    # type: (str, str) -> bool
    """
    Should the given word be considered a good context word? I.e. we remove function words, determiners, modifiers...

    (See Nagata pp. 819)
    """
    return all((
        t not in ('DT', 'POS', 'CD', 'PRP', 'PRP$'),
        t not in ('.', ',', ':', '-LRB-', '-LCB-', '-RRB-', '-RCB-', "''", "``"),
        w not in MODIFIERS.COUNT,
        w not in MODIFIERS.MASS,
        w not in MODIFIERS.WHO_KNOWS,
    ))


def assign_target_class(bnp, head):
    # type: (Tree) -> Union['U', 'C']
    """
    labels the given base noun phrase as countable (C) or uncountable (U) (or None if the label cannot be inferred)
    based on the premodifying words.

    (See Nagata pp. 817-18)
    """
    if head.get_label() in ('NNS', 'NNPS'):
        return 'C'

    words_before_head = set([
        lemmatize(node.get_word_form(), tag=node.get_label()).lower() for node in bnp
        if node.is_leaf() and node.order_nb < head.order_nb
    ])
    if words_before_head & MODIFIERS.COUNT:
        if len({'a', 'little'} & words_before_head) == 2:
            return
        else:
            return 'C'
    if words_before_head & MODIFIERS.MASS:
        return 'U'
    if words_before_head & MODIFIERS.WHO_KNOWS:
        return
    return 'U'


def extract_context_for_bnp(bnp, sentence_word_tag_pairs, k, head_lemma):

    np = [lemmatize(w, tag=t).lower() for w, t in bnp.get_word_tag_pairs() if word_good_for_context(w, t)]
    assert head_lemma in np, (np, head_lemma, bnp)
    np.remove(head_lemma)

    left_boundary_idx = bnp.get_leftmost_child().order_nb
    k_minus = [lemmatize(w, tag=t).lower() for w, t in reversed(sentence_word_tag_pairs[:left_boundary_idx]) if word_good_for_context(w, t)][:k]

    right_boundary_idx = bnp.get_rightmost_child().order_nb
    k_plus = [lemmatize(w, tag=t).lower() for w, t in sentence_word_tag_pairs[right_boundary_idx + 1:] if word_good_for_context(w, t)][:k]

    return tuple(np), tuple(k_minus), tuple(k_plus)


def extract_contexts(tree, k=3):
    # type: (Tree) -> List[str, Tuple[Tuple, Tuple, Tuple]]
    word_tag_pairs = tree.get_word_tag_pairs()
    contexts = defaultdict(lambda: defaultdict(list))
    for n in tree:
        if n.get_label() == 'NP':
            try:
                h = n.get_head_collins()
            except IndexError:
                continue
            while not h.is_leaf():
                h = h.get_head_collins()
            head_lemma = lemmatize(h.get_word_form(), tag=h.get_label()).lower()
            if not word_good_for_context(head_lemma, h.get_label()):
                continue

            np, k_minus, k_plus = extract_context_for_bnp(n, word_tag_pairs, k, head_lemma)
            target_class = assign_target_class(n, h)
            if target_class:
                contexts[head_lemma][target_class].append((tuple(np), tuple(k_minus), tuple(k_plus)))

    return contexts


if __name__ == '__main__':
    path = SETTINGS.get('paths', 'dataParsed')
    contexts = {}  # the resulting feature table
    for r, ds, fs in os.walk(path):
        if r in [os.path.join(path, dir_nb) for dir_nb in ('22', '23', '24')]:
            continue
        print r
        ds.sort()
        fs.sort()
        for f in fs:
            f_path = os.path.join(r, f)
            with open(f_path, 'r') as f:
                for l in f:
                    if l:
                        tree = Tree.from_string(l)
                        contexts.update(extract_contexts(tree))

    print len(contexts)
    out_path = SETTINGS.get('paths', 'dataCountability')
    pickle_path = os.path.join(out_path, 'contexts.pkl')
    pickle.dump(contexts, open(pickle_path, 'wb'))
