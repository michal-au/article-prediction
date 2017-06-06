import pytest


@pytest.mark.xfail(reason='nevychazi: je treba zohlednit POS NPs a other DETS')
def test_nonbnps_without_dets(test_forest):
    non_base_nps_with_determiners = []
    for t in test_forest:
        for n in t:
            if n.get_label() == "NP":
                ch_labels = set([ch.get_label() for ch in n.children])
                if not set(["NP", "DT"]) - ch_labels:
                    non_base_nps_with_determiners.append(n)

    for t in non_base_nps_with_determiners:
        print t

    assert not non_base_nps_with_determiners
