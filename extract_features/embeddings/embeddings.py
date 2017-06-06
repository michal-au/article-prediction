def head_form_embeddings(model, head_form):
    try:
        return model[head_form]
    except KeyError:
        return None
