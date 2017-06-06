
def ngrams(bnp, ng):
    leftmost_child = bnp.get_leftmost_child()
    idx = leftmost_child.order_nb
    root = bnp.get_root()
    leaves = root.get_word_tag_pairs()
    leaves = [wtp[0].lower() for wtp in leaves[: idx] if wtp[0].lower() not in ('the', 'a', 'an')]
    if len(leaves) < 2:
        return 0, 0, 0
    a, b = leaves[-2], leaves[-1]
    try:
        c = [w.lower() for w in bnp.get_words() if w.lower() not in ('the', 'a', 'an')][0]
    except IndexError:
        return 0, 0, 0
    s = float(ng[a][b][c]['the'] + ng[a][b][c]['an'] + ng[a][b][c]['a'] + ng[a][b][c]['ZERO'])
    if s:
        return ng[a][b][c]['ZERO']/s, ng[a][b][c]['the']/s, (ng[a][b][c]['a'] + ng[a][b][c]['an'])/s

    return 0, 0, 0
