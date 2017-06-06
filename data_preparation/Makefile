extract-sentences:
	@cd ../.. && \
	python -m code.data_preparation.extract_sentences

tag:
	@cd ../.. && \
	python -m code.data_preparation.tag

parse:
	@cd parse && echo "changing directory to parse" && \
	make parse

parse-repair:
	@cd parse && echo "changing directory to parse" && \
	make parse-repair

move-orig-parses:
	@cd parse && echo "changing directory to parse" && \
	make move-orig-parses

extract-sentences-for-lm:
	@cd ../.. && \
	python -m code.data_preparation.extract_sentences_for_rnnlm

preprocess-billion-benchmark:
	@cd ../.. && \
	python -m code.data_preparation.billion-benchmark.preprocess data/billion_word_benchmark/orig/*
