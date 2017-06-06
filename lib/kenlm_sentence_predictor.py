from Constants import Constants


ARTICLE_OPTIONS = (Constants.article_definite_token, Constants.article_indefinite_bucket_token, '')


def predict_sentence(model, tokens, threshold=None, margin=None):
    already_predicted = []
    for idx, token in enumerate(tokens):
        score = []
        for article_token in ARTICLE_OPTIONS:
            if article_token:
                candidate_sent = ' '.join(already_predicted + [article_token] + tokens[idx:]).strip()
            else:
                candidate_sent = ' '.join(already_predicted + tokens[idx:]).strip()
            score.append(model.perplexity(candidate_sent))

        max_value = min(score)
        if threshold and max_value > threshold:
            already_predicted.append(token)
            continue
        if margin:
            sorted_score = sorted(score, reverse=True)
            if sorted_score[1] > sorted_score[0] - margin*sorted_score[0]:
                # model si je malo jisty, neudelame nic:
                already_predicted.append(token)
                continue
        max_index = score.index(max_value)
        print ' '.join(already_predicted + ["|||"] +tokens[idx:]).strip()
        print score, max_value, max_index, ARTICLE_OPTIONS[max_index]
        if ARTICLE_OPTIONS[max_index]:
           already_predicted.append(ARTICLE_OPTIONS[max_index])
        already_predicted.append(token)
    return already_predicted