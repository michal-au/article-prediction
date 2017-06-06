## ARTICLE GENERATION

Experiments in the article (the/a(n)/0) generation task on WSJ data, conducted as part of my master thesis at the Faculty of Mathematics and Physics, Charles University in Prague (2017).

#### Requirements
* MXPOST tagger http://www.inf.ed.ac.uk/resources/nlp/local_doc/MXPOST.html
* Collins parser http://www.cs.columbia.edu/~mcollins/code.html
* Stanford parser https://nlp.stanford.edu/software/lex-parser.shtml
* Penn Treebank https://catalog.ldc.upenn.edu/ldc99t42
* BNC http://www.natcorp.ox.ac.uk/
* 1-billion-word-benchmark http://www.statmt.org/lm-benchmark/

the corresponding paths need to be set in the `.settings` file
####

#### Data Preparation
Follow the targets in the corresponding `./data_preparation/Makefile` to extract, tag and parse sentences from the WSJ part of the Penn Treebank.

To prepare data for the countability feature, follow the targets in `./data_preparation/bnc/Makefile`
