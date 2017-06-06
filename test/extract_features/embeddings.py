from ...extract_features.embeddings.embeddings import embed_poly


def test_embed_poly_empty():
    assert embed_poly(None) == []


def test_embed_poly():
    assert embed_poly([1, 2, 3, 4]) == [1, 2, 3, 4, 4, 6, 8, 9, 12, 16]
