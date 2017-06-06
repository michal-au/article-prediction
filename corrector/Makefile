eval-manual-fillin:
	@cd ../.. && \
	python -m code.corrector.eval_manual

eval-manual-print-all-together:
	@cd ../.. && \
	python -m code.corrector.eval_manual --all-together

eval-manual-fillin-parse-only:
	@cd ../.. && \
	python -m code.corrector.eval_manual --parse-fillin-orig-noarticles

eval-penn-test:
	@cd ../.. && \
	python -m code.corrector.eval_penn

eval-penn-print-all-together:
	@cd ../.. && \
	python -m code.corrector.eval_penn --all-together

eval-show-automatic-parse-candidates-for-prediction:
	@cd ../.. && \
	python -m code.corrector.eval_manual --show-automatic-parse-candidates-for-prediction

predict-manual-by-xgb:
	@cd ../.. && \
	python -m code.corrector.predict_by_classifier --model-type xgboost --input data/corrector/manual/fill_in_orig.txt --output data/corrector/manual/fill_in/solution_XGB.txt

predict-manual-by-xgb-bnc:
	@cd ../.. && \
	python -m code.corrector.predict_by_classifier --model-type bnc-xgboost --input data/corrector/manual/fill_in_orig.txt --output data/corrector/manual/fill_in/solution_XGB_BNC400_000.txt

predict-manual-by-logreg:
	@cd ../.. && \
	python -m code.corrector.predict_by_classifier --model-type xgboost --input data/corrector/manual/fill_in_orig.txt --output data/corrector/manual/fill_in/solution_LOGREG.txt

predict-manual-by-lm:
	@cd ../.. && \
	python -m code.corrector.predict_by_classifier --model-type kenlm --input data/corrector/manual/fill_in_orig.txt --output data/corrector/manual/fill_in/solution_LM50.txt


predict-penn-by-lm:
	@cd ../.. && \
	python -m code.corrector.predict_by_classifier --model-type kenlm --input data/corrector/penn/penn_test_raw_orig.txt --output data/corrector/penn/solution_LM_0_0.txt

predict-penn-by-xgb:
	@cd ../.. && \
	cat data/raw/23/wsj_23* | sed 's|\\/|\/|g' > penn_test_section_raw.tmp && \
	python -m code.corrector.predict_by_classifier --model-type xgboost --input penn_test_section_raw.tmp --output data/corrector/penn/solution_XGB.txt && \
	rm penn_test_section_raw.tmp

predict-penn-by-logreg:
	@cd ../.. && \
	cat data/raw/23/wsj_23* | sed 's|\\/|\/|g' > penn_test_section_raw.tmp && \
	python -m code.corrector.predict_by_classifier --model-type logreg --input penn_test_section_raw.tmp --output data/corrector/penn/solution_LOGREG.txt && \
	rm penn_test_section_raw.tmp