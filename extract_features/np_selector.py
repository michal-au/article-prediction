def penn_selector(n):
    if n.get_label() != "NP":
        return False
    if "NP" not in [ch.get_label() for ch in n.children]:
        return True
    np_children = [ch for ch in n.children if ch.get_label() == "NP"]
    return all(['POS' in [chch.get_label() for chch in np_child.children] for np_child in np_children])


def collins_selector(n):
    return n.get_label() == 'NPB'
