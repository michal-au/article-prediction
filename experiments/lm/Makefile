estimate-threshold-and-margin:
	@cd ../../.. && \
	cat data/raw/22/wsj_22* | sed 's|\\/|\/|g' > penn_heldout_section_raw.tmp && \
	python -m code.experiments.lm.estimate_cutoff penn_heldout_section_raw.tmp && \
	rm penn_heldout_section_raw.tmp


eval-lm:
	@cd ../../.. && \
	cat data/raw/23/wsj_23* | sed 's|\\/|\/|g' > penn_test_section_LM_TEST.tmp && \
	python -m code.experiments.lm.eval_lm --threshold 0 --margin 0 && \
	rm penn_test_section_LM_TEST.tmp