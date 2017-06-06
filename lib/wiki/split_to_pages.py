"""
Split the large xml file into pages
"""

import os
import sys
from ..utils import read_settings


if len(sys.argv) < 2:
    raise NameError("No file given!")

PAGES_PER_FILE = 10000
SETTINGS = read_settings()
OUT_PATH = SETTINGS.get("path", "dataWiki")
IN_PATH = os.path.join(OUT_PATH, 'enwiki-latest-pages-articles.xml')

with open(IN_PATH, 'r') as f:
    text = False
    counter = 0
    out_file = open(os.path.join(OUT_PATH, 'raw', str(counter)), 'w+')
    for l in f:
        if "<text" in l:
            text = True
        if "#redirect" in l.lower():
            text = False
        if text:
            print >>out_file, l
        if "</text" in l:
            if text:
                counter += 1
                if counter >= PAGES_PER_FILE:
                    out_file = open(os.path.join(OUT_PATH, "raw", str(counter)), 'w+')
            text = False
