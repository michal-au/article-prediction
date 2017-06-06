from ...lib.wordnet import lemmatize
from ...lib.utils import read_settings
from ...lib.Tree import Tree
from extract_contexts import extract_context_for_bnp, word_good_for_context
from decision_lists import TEMPLATE_NAMES, DEFAULT_NAME

SETTINGS = read_settings()


def rule_applies(rule, contexts):
    # type: (List, Dict) -> bool
    return rule[0] == DEFAULT_NAME or rule[1] in contexts[rule[0]]


def _attach_rules_with_the_same_score(score, i, decision_list, applied_rules, contexts):
    # type: (int, int, List[Tuple], List[Tuple]) -> (List[Tuple], int)
    """
    from the list of available rules, the same-ranking rules are appended to the list of selected rules
    :param score: log likelihood of the rules we are interested in
    :param i: point in the list from which we search for the rules
    :param decision_list: list of available rules for the given noun phrase
    :param applied_rules: list of selected rules
    :param applied_rules: dict of contexts of the given base np
    :return: list of selected rules, the end point of the search in the decision list
    """
    while i < len(decision_list):
        next_rule = decision_list[i]
        if next_rule[3] < score:
            break
        if rule_applies(next_rule, contexts):
            applied_rules.append(next_rule)
        i += 1
    return applied_rules, i


def _get_countability_score(rules):
    # type: (List[Tuple]) -> int
    """
    for the given list of rules, this returns the difference between the number of 'C' and 'U' rules, (i.e the rules
    that label the phrase as countable or uncountable)
    """
    score = 0
    for r in rules:
        if r[2] == 'C':
            score += 1
        else:
            score -= 1
    return score


def countability(bnp, decision_lists, sentence_word_tag_pairs):
    # type: (Tree, Dict, List[Tuple[str, str]]) -> Union['C', 'U']
    """
    the given noun phrase is labelled as un/countable based on the list of context rules
    :param bnp: base noun phrase
    :param decision_lists: list of context rules (sorted from the most to the least important)
    :param sentence_word_tag_pairs: words with their corresponding POS tags from the whole sentence
    (for context extraction)
    :return: 'C' or 'U'
    """
    h = bnp.get_head_collins()
    while not h.is_leaf():
        h = h.get_head_collins()
    head_lemma = lemmatize(h.get_word_form(), tag=h.get_label()).lower()

    # TODO: maybe try to tag it as if this was the training data for the decision lists?
    if not word_good_for_context(head_lemma, h.get_label()) or not decision_lists.get(head_lemma):
        return

    contexts = {
        TEMPLATE_NAMES[i]: context for i, context in enumerate(extract_context_for_bnp(
            bnp, sentence_word_tag_pairs, 3, head_lemma
        ))
    }

    for idx, rule in enumerate(decision_lists[head_lemma]):
        if rule_applies(rule, contexts):
            decision_list = decision_lists[head_lemma]
            score = rule[3]
            applied_rules, i = _attach_rules_with_the_same_score(score, idx + 1, decision_list, [rule], contexts)

            countability_score = _get_countability_score(applied_rules)
            while countability_score == 0:
                if DEFAULT_NAME == applied_rules[-1][0]:
                    return applied_rules[-1][2]
                assert i < len(decision_list)  # each list must end with a default rule
                if not rule_applies(decision_list[i], contexts):
                    i += 1
                    continue
                next_rule = decision_list[i]
                applied_rules = [next_rule]
                applied_rules, i = _attach_rules_with_the_same_score(next_rule[3], i + 1, decision_list, applied_rules, contexts)
                countability_score = _get_countability_score(applied_rules)

            return 'C' if countability_score > 0 else 'U'

    return None


def countability_with_rules_used(bnp, decision_lists, sentence_word_tag_pairs):
    # type: (Tree, Dict, List[Tuple[str, str]]) -> Union['C', 'U']
    """
    the given noun phrase is labelled as un/countable based on the list of context rules
    :param bnp: base noun phrase
    :param decision_lists: list of context rules (sorted from the most to the least important)
    :param sentence_word_tag_pairs: words with their corresponding POS tags from the whole sentence
    (for context extraction)
    :return: 'C' or 'U'
    """
    h = bnp.get_head_collins()
    while not h.is_leaf():
        h = h.get_head_collins()
    head_lemma = lemmatize(h.get_word_form(), tag=h.get_label()).lower()

    # TODO: maybe try to tag it as if this was the training data for the decision lists?
    if not word_good_for_context(head_lemma, h.get_label()):
        return



    contexts = {
        TEMPLATE_NAMES[i]: context for i, context in enumerate(extract_context_for_bnp(
            bnp, sentence_word_tag_pairs, 3, head_lemma
        ))
    }

    for idx, rule in enumerate(decision_lists[head_lemma]):
        if rule_applies(rule, contexts):
            decision_list = decision_lists[head_lemma]
            score = rule[3]
            applied_rules, i = _attach_rules_with_the_same_score(score, idx + 1, decision_list, [rule], contexts)

            countability_score = _get_countability_score(applied_rules)
            while countability_score == 0:
                if DEFAULT_NAME == applied_rules[-1][0]:
                    return applied_rules[-1][2]
                assert i < len(decision_list)  # each list must end with a default rule
                if not rule_applies(decision_list[i], contexts):
                    i += 1
                    continue
                next_rule = decision_list[i]
                applied_rules = [next_rule]
                applied_rules, i = _attach_rules_with_the_same_score(next_rule[3], i + 1, decision_list, applied_rules, contexts)
                countability_score = _get_countability_score(applied_rules)

            return 'C' if countability_score > 0 else 'U'

    return None


def referent_with_countability(bnp, decision_lists, sentence_word_tag_pairs, referent):
    if not referent:
        return 'n'
    c = countability(bnp, decision_lists, sentence_word_tag_pairs)
    if not c:
        return 'y'
    return 'y'+c
