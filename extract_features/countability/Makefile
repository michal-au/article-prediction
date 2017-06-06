extract-contexts:
	@cd ../../.. && \
	python -m code.extract_features.countability.extract_contexts

extract-decision_lists:
	@cd ../../.. && \
	python -m code.extract_features.countability.decision_lists

delete:
	rm ../../../data/countability/contexts.pkl ../../../data/countability/decision_lists.pkl

countability: extract-contexts extract-decision_lists

