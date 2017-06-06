from __future__ import division
from ....extract_features.additional_methods.predicate import main_verb_form, subject_verb_form, object_verb_form
from ....lib.Tree import Tree


def _get_first_bnp(t):
    bnp = None
    for n in t:
        if n.get_label() == 'NPB':
            bnp = n
            break
    assert bnp
    return bnp


def test_main_verb_form():
    t = Tree.from_string('(S~will~2~2 (NP-A~Vinken~2~1 (NPB~Vinken~2~2 (NNP Pierre) (NNP Vinken) (PUNC, ,) ) (ADJP~old~2~2 (NPB~years~2~2 (CD 61) (NNS years) ) (JJ old) (PUNC, ,) ) ) '
                         '(VP~will~2~1 (MD will) (VP-A~join~4~1 (VB join) (NPB~board~2~2 (DT the) (NN board) )'
                         '(PP~as~2~1 (IN as) (NPB~director~3~3 (DT a) (JJ nonexecutive) (NN director) ) )'
                         '(NPB~Nov.~2~1 (NNP Nov.) (CD 29) (PUNC. .) ) ) ) )')
    assert main_verb_form(_get_first_bnp(t)) == 'will|LEFT'

    t = Tree.from_string('(S~will~2~2 (NP-A~Vinken~2~1 (NP~Vinken~2~2 (NNP Pierre) (NNP Vinken) (PUNC, ,) ) (ADJP~old~2~2 (NP~years~2~2 (CD 61) (NNS years) ) (JJ old) (PUNC, ,) ) ) '
                         '(VP~will~2~1 (MD will) (VP-A~join~4~1 (VB join) (NP~board~2~2 (DT the) (NN board) )'
                         '(PP~as~2~1 (IN as) (NPB~director~3~3 (DT a) (JJ nonexecutive) (NN director) ) )'
                         '(NPB~Nov.~2~1 (NNP Nov.) (CD 29) (PUNC. .) ) ) ) )')
    print _get_first_bnp(t)
    assert main_verb_form(_get_first_bnp(t)) == 'will|RIGHT'


def test_subject_verb_form():
    t = Tree.from_string('(S~will~2~2 (NP-A~Vinken~2~1 (NPB~Vinken~2~2 (NNP Pierre) (NNP Vinken) (PUNC, ,) ) (ADJP~old~2~2 (NPB~years~2~2 (CD 61) (NNS years) ) (JJ old) (PUNC, ,) ) ) '
                         '(VP~will~2~1 (MD will) (VP-A~join~4~1 (VB join) (NPB~board~2~2 (DT the) (NN board) )'
                         '(PP~as~2~1 (IN as) (NPB~director~3~3 (DT a) (JJ nonexecutive) (NN director) ) )'
                         '(NPB~Nov.~2~1 (NNP Nov.) (CD 29) (PUNC. .) ) ) ) )')
    assert subject_verb_form(_get_first_bnp(t)) == 'will join'


def test_subject_verb_form2():
    t = Tree.from_string('(TOP~was~1~1 (S~was~3~3 (PP~In~2~1 (IN In) (NPB~trading~4~4 (JJ late) (NNP New) (NNP York) (NN trading)))'
                         '(NPB~currency~2~2 (DT the) (NN currency)) (VP~was~2~1 (VBD was) (VP~quoted~3~1 (VBN quoted)'
                         '(PP~at~2~1 (IN at) (NP~marks~3~1 (NPB~marks~2~2 (CD 1.8355) (NNS marks)) (CC and) (NPB~yen~2~2 (CD 141.45) (NN yen) (, ,))))'
                         '(PP~compared~2~1 (VBN compared) (PP~with~2~1 (IN with) (NP~marks~3~1 (NPB~marks~2~2 (CD 1.8470) (NNS marks))'
                         '(CC and) (NP~yen~2~1 (NPB~yen~2~2 (CD 141.90) (NN yen)) (NPB~Monday~1~1 (NNP Monday) (. .))))))))))' )
    assert subject_verb_form(_get_first_bnp(t)) == 'be quote'


def test_object_verb_form():
    t = Tree.from_string('(S (VP (TO to) (VP (VB inject) (NPB (JJ Big) (NNS Funds)))))')
    assert object_verb_form(_get_first_bnp(t)) == 'to inject'

    t = Tree.from_string('(S (VP (VB See) (NP (NPB (JJ related) (NN story)))))')
    assert object_verb_form(_get_first_bnp(t)) == 'see'

    # with preposition:
    t = Tree.from_string('(S (VP (TO to) (VP (VB get) (PP (IN into) (NPB (DT a) (NN debate))))))')
    assert object_verb_form(_get_first_bnp(t)) == 'to get into'

    t = Tree.from_string('(NP (NP (NN stock) (NNS prices)) (VP (VBN quoted) (PP (IN on) (NPB (DT the) (NNP Big) (NNP Board)))))')
    assert object_verb_form(_get_first_bnp(t)) == 'quote on'
