benchmark_data_path := $(HOME)/diplomka/data/1-billion-word-language-modeling-benchmark-r13output/
kenlm_model_path := $(HOME)/diplomka/data/model/lang-model

postprocess-features:
	@cd ../.. && \
	python -m code.experiments.df_transform.postproces_and_store_features_locally

lee_tune:
	@cd ../.. && \
	python -m code.experiments.lee_tuning.tune

lee-train-from-file:
	@cd ../.. && \
	python -m code.experiments.lee_tuning.train_from_file code/logs/experiments/model_results/penn/instructions.csv

rf-train-from-file:
	@cd ../.. && \
	python -m code.experiments.lee_tuning.train_from_file code/logs/experiments/model_results/penn/instructions-rf.csv

lee_tune_console_only:
	@cd ../.. && \
	python -m code.experiments.lee_tuning.tune console_only

lee_tune_console_only_cross_valid:
	@cd ../.. && \
	python -m code.experiments.lee_tuning.tune console_only cross_valid

kenlm-train:
	@cd ../.. && \
	python -m code.experiments.lm.kenlm

kenlm-eval:
	@cd ../.. && \
	python -m code.experiments.lm.kenlm_eval

kenlm-train-ggl-5-without-my-preprocessing: ggl5-traindata-nopreproc
	$(HOME)/kenlm/build/bin/lmplz -S 50% -o 5 < ggl5-traindata-nopreproc > ggl5-model.tmp
	$(HOME)/kenlm/build/bin/build_binary ggl5-model.tmp $(kenlm_model_path)/kenlm-ggl-5-without-my-preprocessing

kenlm-eval-ggl-5-without-my-preprocessing:
	$(HOME)/kenlm/build/bin/query -v summary $(kenlm_model_path)/kenlm-ggl-5-without-my-preprocessing < $(benchmark_data_path)/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050

kenlm-train-ggl-5-with-nbs-cls3: ggl5-traindata-nbs-cls3
	$(HOME)/kenlm/build/bin/lmplz -S 50% -o 5 < ggl5-traindata-nbs-cls3 > ggl5-nbs-cl3-model.tmp
	$(HOME)/kenlm/build/bin/build_binary ggl5-nbs-cl3-model.tmp $(kenlm_model_path)/kenlm-ggl-5-nbs-cls3

kenlm-train-ggl-5-with-nbs-cls4: ggl5-traindata-nbs-cls4
	$(HOME)/kenlm/build/bin/lmplz -S 50% -o 5 < ggl5-traindata-nbs-cls4 > ggl5-nbs-cls4-model.tmp
	$(HOME)/kenlm/build/bin/build_binary ggl5-nbs-cls4-model.tmp $(kenlm_model_path)/kenlm-ggl-5-nbs-cls4

ggl5-traindata-nopreproc:
	cat $(benchmark_data_path)/training-monolingual.tokenized.shuffled/* >> ggl5-traindata-nopreproc

ggl5-traindata-nbs-cls3:
	@cd ../.. && python -m code.data_preparation.billion-benchmark.preproces_sentence --nbs --cls3 $(benchmark_data_path)/training-monolingual.tokenized.shuffled/* >> code/experiments/ggl5-traindata-nbs-cls3

ggl5-traindata-nbs-cls4:
	@cd ../.. && python -m code.data_preparation.billion-benchmark.preproces_sentence --nbs $(benchmark_data_path)/training-monolingual.tokenized.shuffled/* >> code/experiments/ggl5-traindata-nbs-cls4


kenlm-clean:
	rm -f ggl5-traindata-nopreproc ggl5-model.tmp ggl5-nbs-cl3-model.tmp ggl5-traindata-nbs-cls3 ggl5-traindata-nbs-cls4
