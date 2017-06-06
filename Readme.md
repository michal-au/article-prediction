## ARTICLE GENERATION

Experiments in the article (the/a(n)/0) generation task on WSJ data, conducted as part of my master thesis at the Faculty of Mathematics and Physics, Charles University in Prague (2017).

#### Requirements
* MXPOST tagger http://www.inf.ed.ac.uk/resources/nlp/local_doc/MXPOST.html
* Collins parser http://www.cs.columbia.edu/~mcollins/code.html
* Stanford parser https://nlp.stanford.edu/software/lex-parser.shtml
* Penn Treebank https://catalog.ldc.upenn.edu/ldc99t42
* BNC http://www.natcorp.ox.ac.uk/
* 1-billion-word-benchmark http://www.statmt.org/lm-benchmark/
* KenLM language model https://kheafield.com/code/kenlm/
* XGBoost https://github.com/dmlc/xgboost

the corresponding paths need to be set in the `.settings` file
####

#### Data Preparation
Follow the targets in the corresponding `./data_preparation/Makefile` to extract, tag and parse sentences from the WSJ part of the Penn Treebank.

To prepare data for the countability feature, follow the targets in `./data_preparation/bnc/Makefile`

To prepare data for the language model feature, execute the `preprocess-billion-benchmark` target in `./data_preparation/Makefile`

#### Feature Extraction
To extract the features used by the classifiers, first prepare decision lists for the countability feature: `cd extract_features/countability_bnc/ && make countability-bnc`. Then train a language model on the 1-billion-word benchmark: `cd experiments && make kenlm-train-ggl-5-with-nbs-cls3`. Then, to extract all the features for the wsj corpus run `extract-penn-features` target in `extract_features/Makefile`. Finally, to prepare the extracted features for machine learning algorithms, postprocess them by the `postprocess-features` target in `experiments/Makefile`.

#### Experiments
To train and evaluate logistic regression models: `cd experiments && make lr-train-from-file`. The model expects an instruction file specifying the parameters of the model, such as the regularization parameter and the features to use (`logs/experiments/model_results/penn/instructions.csv`), for the format of the file, see `lib.train_model_on_postprocessed_features.train_model`. 

For experimenting with the gradient boosted tree models, follow the jupyter notebook in `experiments/notebook_lee_tuning/XGBoost.ipynb`
