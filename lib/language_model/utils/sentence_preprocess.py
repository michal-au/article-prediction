from ...Constants import Constants


def replace_indef_tokens_by_bucket(token):
    if token.lower() in Constants.article_indefinite_tokens:
        return Constants.article_indefinite_bucket_token
    else:
        return token


def process_nbs(token):
    try:
        float(token)
        token = "<number>"
    except ValueError:
        pass
    return token


def preprocess_sent_for_zero_lc_nb(tokens):
    """
    Used for feature extraction, when we want lang model prediction as feature
    """
    sent = [Constants.no_article_token]
    for token in tokens:
        token = token.lower()
        token = process_nbs(token)
        if token in Constants.article_tokens:
            continue
        sent.append(token)
        sent.append(Constants.no_article_token)
    return sent


def preprocess_sent_for_lc_nb(tokens):
    """
    Used for feature extraction, when we want lang model prediction as feature
    """
    sent = [Constants.no_article_token]
    for token in tokens:
        token = token.lower()
        token = process_nbs(token)
        if token in Constants.article_tokens:
            continue
        sent.append(token)
    return sent
