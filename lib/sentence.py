from Constants import Constants


class SentenceStateTracker:

    def __init__(self, tokens, zero_filled=False):
        self.original = [t for t in tokens]
        self.so_far_predicted = []
        self.last_prediction_idx = 0
        self.zero_filled = zero_filled

    def get_prediction_candidates(self, bnp):
        current_idx = bnp.get_leftmost_child().order_nb
        if self.zero_filled:
            current_idx *= 2
        previous_words = self.so_far_predicted + self.original[self.last_prediction_idx:current_idx]
        following_words = self.original[current_idx+1:] if self.zero_filled else self.original[current_idx:]
        return (
            ' '.join(previous_words + [Constants.article_indefinite_bucket_token] + following_words),
            ' '.join(previous_words + [Constants.article_definite_token] + following_words),
            ' '.join(previous_words + ([Constants.no_article_token] if self.zero_filled else []) + following_words),
        )

    def record_prediction(self, bnp, token):
        idx = bnp.get_leftmost_child().order_nb
        if self.zero_filled:
            idx *= 2
        self.so_far_predicted += self.original[self.last_prediction_idx:idx]
        if token:
            self.so_far_predicted.append(token)
        self.last_prediction_idx = idx + 1 if self.zero_filled else idx

    @property
    def candidate_tokens(self):
        return (
            Constants.article_indefinite_bucket_token,
            Constants.article_definite_token,
            Constants.no_article_token
        )
