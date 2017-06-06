import pytest

from ....extract_features.countability.extract_contexts import extract_contexts, assign_target_class
from ....lib.Tree import Tree


def test_extract_contexts():
    t = Tree.from_string('(TOP~is~1~1 (S~is~2~2'
                         '(NPB~Vinken~2~2 (NNP Mr.) (NNP Vinken)) (VP~is~2~1 (VBZ is) (NP~chairman~2~1 (NPB~chairman~1~1 (NN chairman))'
                         '(PP~of~2~1 (IN of) (NP~N.V.~2~1 (NPB~N.V.~2~2 (NNP Elsevier) (NNP N.V.) (, ,))'
                         '(NPB~group~4~4 (DT the) (JJ Dutch) (NN publishing) (NN group) (. .))))))))')
    contexts = extract_contexts(t)
    assert len(contexts) == 3
    assert 'vinken' in contexts, contexts.keys()
    assert 'U' in contexts['vinken']
    assert contexts['vinken']['U'] == [(('mr.',), (), ('be', 'chairman', 'of'))]

    assert 'n.v.' in contexts, contexts.keys()
    assert 'U' in contexts['n.v.']
    assert contexts['n.v.']['U'] == [(('elsevier',), ('of', 'chairman', 'be'), ('dutch', 'publishing', 'group'))]

    assert 'chairman' in contexts, contexts.keys()
    assert 'U' in contexts['chairman']
    assert contexts['chairman']['U'] == [((), ('be', 'vinken', 'mr.'), ('of', 'elsevier', 'n.v.'))]


def test_extract_contexts_without_target_class():
    t = Tree.from_string('(VP (NPB (DT the) (ADJ perfect) (NN example) ) (PP (P of) (NPB (DT that) (NN stuff)) ))')
    contexts = extract_contexts(t)
    assert len(contexts) == 0


@pytest.mark.parametrize('bnp_string, target_class', [
    ('(NPB (DT the) (ADJ perfect) (NN example) )', None),
    ('(NPB (DT the) (ADJ perfect) (NNS examples) )', 'C'),
    ('(NPB (DT another) (NN example) )', 'C'),
    ('(NPB~coherence~3~3 (DT a) (JJ little) (NN coherence))', None),
    ('(NPB (DT a) (JJ little) (NN dog))', None),
    ('(NPB (DT a) (JJ small) (NN dog))', 'C'),
    ('(NPB (DT an) (JJ oval) (NN dog))', 'C'),
    ('(NPB (JJ My) (NN dog))', None),
    ('(NPB (NN experience))', 'U'),
    ('(NPB (JJ all) (NN flour))', 'U'),
    ('(NPB (JJ enough) (NN flour))', 'U'),
    ('(NPB~money~2~2 (JJ little) (NN money)) (VP~left~2~1 (VBN left) (S~to~1~1 (VP~to~2~1 (TO to) (VP~spend~2~1 (VB spend) (PP~on~2~1 (IN on) (NPB~products~2~2 (JJ other) (NNS products) (. .))', 'U'),
])
def test_assign_target_class(bnp_string, target_class):
    bnp = Tree.from_string(bnp_string)
    head = bnp.get_head_collins()
    assert assign_target_class(bnp, head) == target_class
