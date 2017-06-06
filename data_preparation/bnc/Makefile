#These steps should be taken in the given order!

extract-text:
	cd ../../../data && \
	mkdir bnc-raw && \
	rsync -a -f'+ */' -f'- *' bnc/2554/download/Texts/ bnc-raw && \
    find ./bnc/2554/download/Texts -type f -exec sh -c 'xsltproc ../code/data_preparation/bnc/justTheWords.xsl $$1 | tr -s " " > bnc-raw/$${1#./bnc/2554/download/Texts/} ' _ {} \; && \
    echo OK


remove-empty-files:
	cd ../../../data && \
	find bnc-raw -type f -size 1 -delete && \
	find bnc-raw -depth -type d -empty -delete && \
	echo OK


# put aside 8 articles for human evaluation
separate-manual-test-data:
	cd ../../.. && \
	python -m code.data_preparation.bnc.separate_manual_test_data && \
	echo OK


# separate data to train - heldout - test
separate-data:
	cd ../../.. && \
	python -m code.data_preparation.bnc.separate_data && \
	echo OK


parse-train:
	cd ../../.. && \
	python -m code.data_preparation.bnc.parse && \
	echo OK
