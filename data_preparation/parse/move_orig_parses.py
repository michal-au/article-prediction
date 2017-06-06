# This script takes the original parsed files from the test part of the Penn corpus and converts them
# to the format that is used in the automatically parsed data, that is one tree per line
# This is just for convenience so that features can be extracted in the same way from any data we care about

import os
from ...lib import utils, Tree


def format_orig_parses(fPath):
    trees = Tree.Tree.from_file(fPath)
    for t in trees:
        t.delete_nodes_by_value('-NONE-')
        for node in t:
            if len(node.children) > 0 and not node.val.startswith('-'):
                # discarding the additional info from tags, such as: "-TMP" from "NP-TMP"
                # at the same time preserving -LRB-, -RRB- tags
                node.val = node.val.split('-')[0]

    return [t.to_string() for t in trees]


if __name__ == '__main__':
    settings = utils.read_settings()
    path = settings.get('paths', 'dataOrig')
    out_path = settings.get('paths', 'dataParsedOrig')
    for r, ds, fs in sorted(os.walk(path)):
        if r.endswith('23'):
            for f in sorted(fs):
                l = format_orig_parses(os.path.join(r, f))
                utils.print_list_to_file(l, os.path.join(out_path, '23', f))
