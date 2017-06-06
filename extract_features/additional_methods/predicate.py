import typing
#from ...lib.Tree import Tree
from ...lib.wordnet import lemmatize


def main_verb_form(bnp):
    s = bnp
    while not s.is_root():
        if s.get_label() == 'S':
            break
        s = s.parent
    if not s.get_label() == 'S':
        return None

    bnp_child_nb = 1
    for ch in s.children:
        if bnp in ch:
            break
        if not ch.get_label().startswith('PUNC'):
            bnp_child_nb += 1

    position = 'LEFT'
    try:
        if bnp_child_nb >= int(s.val.split('~')[-1]):
            position = 'RIGHT'
    except ValueError:
        # test data set... nejsou parsovany collins-parserem....
        print s
        print '>>>>>>'
        print bnp

    return s.get_label() + '|' + position  # TODO: wtf???? check the score


def subject_verb_form(bnp):
    if bnp.is_root():
        return
    parent = bnp.parent

    while not parent.is_root() and parent.get_label() in ('NP', 'NPB', 'FRAG'):  # TODO: exclude Frag?
        parent = parent.parent

    if parent.parent and parent.parent.get_label().startswith('S'):
        parent = parent.parent
        if len(parent.children) > 1:
            vp = None
            bnp_idx = 0
            for idx, ch in enumerate(parent.children):
                if ch == bnp:
                    bnp_idx = idx
            for ch in parent.children[bnp_idx:]:
                if ch.get_label().startswith('VP'):
                    vp = ch
                    break
            if not vp:
                for ch in reversed(parent.children[:bnp_idx]):
                    if ch.get_label().startswith('VP'):
                        vp = ch
                        break
            if vp:
                predicate = _process_vp(vp, [])
                return ' '.join(predicate)


def object_verb_form(bnp):  # TODO testit!
    if bnp.is_root():
        return
    parent = bnp.parent

    # todo: attach PP head???
    while not parent.is_root() and parent.get_label() in ('NP', 'NPB', 'FRAG'):  # TODO: exclude Frag?
        parent = parent.parent

    preposition = ''
    if parent.get_label().startswith('PP') and parent.parent:
        preposition = parent.children[0].get_word_form()
        parent = parent.parent

    if parent.get_label().startswith('VP'):
        predicate = _process_vp(parent, [])
        predicate = ' '.join(predicate)
        return predicate + ' ' + preposition if preposition else predicate


def _process_vp(vp, res=[]):
    verb = [ch for ch in vp.children if ch.get_label() in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')]
    if verb:
        verb = verb[0]
        res.append(lemmatize(verb.get_word_form(), tag=verb.get_label()))
    verb_phrase = [ch for ch in vp.children if ch.get_label().startswith('VP')]  # VP-A (see collins parser readme)
    if verb_phrase:
        verb_phrase = verb_phrase[0]
        res = _process_vp(verb_phrase, res)
    return res


def _process_vp_bottom_up(vp, res=[]):
    if vp.parent and vp.parent.get_label().startswith('VP'):
        res = _process_vp_bottom_up(vp.parent, res)
    verbs = [ch for ch in vp.children if ch.get_label() in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'TO')]
    if verbs:
        res.extend([lemmatize(verb.get_word_form(), tag=verb.get_label()).lower() for verb in verbs])
    return res


def pp_object_form(bnp):  # TODO testit!
    if bnp.is_root():
        return
    parent = bnp.parent

    while not parent.is_root() and parent.get_label() in ('NP', 'NPB', 'FRAG'):  # TODO: exclude Frag?
        parent = parent.parent

    if parent.get_label().startswith('PP'):
        return parent.get_leftmost_child().get_word_form().lower()


def subject_position(bnp):  # TODO testit!
    if bnp.is_root():
        return
    parent = bnp.parent

    while not parent.is_root() and parent.get_label() in ('NP', 'NPB', 'FRAG'):  # TODO: exclude Frag?
        parent = parent.parent

    if parent.parent and parent.parent.get_label().startswith('S'):
        parent = parent.parent
        if [ch for ch in parent.children if ch.get_label().startswith('VP')]:
            return True
    return False