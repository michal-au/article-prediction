

def score_by_kenlm(model, bnp, sst, prob=True):
    # (model, Tree, SentenceStateTracker)
    if prob:
        scores = [model.score(s) for s in sst.get_prediction_candidates(bnp)]
    else:
        scores = [-model.perplexity(s) for s in sst.get_prediction_candidates(bnp)]

    # zapis predikce do vety (pro rozhodovani na dalsich mistech ve vete)
    max_score = max(scores)
    predicted_token = next((token for idx, token in enumerate(sst.candidate_tokens) if scores[idx] == max_score))
    sst.record_prediction(bnp, predicted_token)

    return predicted_token
