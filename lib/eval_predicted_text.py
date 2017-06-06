import nltk

ARTICLE_TOKENS = ('a', 'an', 'a/an', 'the')


def eval_predicted_text(orig_sent, predict_sent, tokenize=True, cv=None):
    swaps, insertions, deletions, correct = 0, 0, 0, 0
    error_repr = []
    if tokenize:
        orig_sent, predict_sent = nltk.word_tokenize(orig_sent), nltk.word_tokenize(predict_sent)

    orig_idx, predict_idx = 0, 0
    while orig_idx < len(orig_sent) or predict_idx < len(predict_sent):
        orig_word, predict_word = orig_sent[orig_idx].lower(), predict_sent[predict_idx].lower()
        if orig_word == predict_word or (orig_word in ('a', 'an', 'a/an') and predict_word in ('a', 'an', 'a/an')):
            orig_idx += 1
            predict_idx += 1
            if orig_word in ARTICLE_TOKENS:
                correct += 1
                if cv is not None:
                    if orig_word in ('a', 'an', 'a/an'):
                        orig_word = 'a/an'
                    if predict_word in ('a', 'an', 'a/an'):
                        predict_word = 'a/an'
                    cv[orig_word][predict_word] += 1
            error_repr.append(orig_word)
            continue
        if orig_word in ARTICLE_TOKENS and predict_word in ARTICLE_TOKENS:
            print orig_word, predict_word
            swaps += 1
            orig_idx += 1
            predict_idx += 1
            if cv is not None:
                if orig_word in ('a', 'an', 'a/an'):
                    orig_word = 'a/an'
                if predict_word in ('a', 'an', 'a/an'):
                    predict_word = 'a/an'
                cv[orig_word][predict_word] += 1
            error_repr.append('[{}/{}]'.format(predict_word, orig_word))
            continue
        if orig_word in ARTICLE_TOKENS:
            deletions += 1
            orig_idx += 1
            if cv is not None:
                if orig_word in ('a', 'an', 'a/an'):
                    orig_word = 'a/an'
                cv[orig_word]['ZERO'] += 1
            error_repr.append('[-/{}]'.format(orig_word))
        elif predict_word in ARTICLE_TOKENS:
            insertions += 1
            predict_idx += 1
            if cv is not None:
                if predict_word in ('a', 'an', 'a/an'):
                    predict_word = 'a/an'
                cv['ZERO'][predict_word] += 1
            error_repr.append('[{}/-]'.format(predict_word))
        else:
            print(u"!!!: orig: >{}<, predict: >{}<".format(orig_word, predict_word))
            print error_repr
            raise ValueError(u"!!!: orig: {}, predict: {}".format(orig_word, predict_word))

    if not cv:
        return swaps, insertions, deletions, correct, error_repr
    else:
        return swaps, insertions, deletions, correct, error_repr, cv