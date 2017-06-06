benchmark_data_path := $(HOME)/diplomka/data/1-billion-word-language-modeling-benchmark-r13output/
kenlm_model_path := $(HOME)/diplomka/data/model/lang-model

postprocess-features:
	@cd ../.. && \
	python -m code.experiments.df_transform.postproces_and_store_features_locally

lr-train-from-file:
	@cd ../.. && \
	python -m code.experiments.lee_tuning.train_from_file code/logs/experiments/model_results/penn/instructions.csv

kenlm-train-ggl-5-with-nbs-cls3: ggl5-traindata-nbs-cls3
	$(HOME)/kenlm/build/bin/lmplz -S 50% -o 5 < ggl5-traindata-nbs-cls3 > ggl5-nbs-cl3-model.tmp
	$(HOME)/kenlm/build/bin/build_binary ggl5-nbs-cl3-model.tmp $(kenlm_model_path)/kenlm-ggl-5-nbs-cls3

ggl5-traindata-nbs-cls3:
	@cd ../.. && python -m code.data_preparation.billion-benchmark.preproces_sentence --nbs --cls3 $(benchmark_data_path)/training-monolingual.tokenized.shuffled/* >> code/experiments/ggl5-traindata-nbs-cls3

kenlm-clean:
	rm -f ggl5-traindata-nbs-cls3
