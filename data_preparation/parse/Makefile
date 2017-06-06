SHELL := /bin/bash


# STEP 1
parse:
	@cd ../../.. && \
	python -m code.data_preparation.parse.parse

get-parsing-errors:
	@cd ../../.. && \
	python -m code.data_preparation.parse.parse_get_errors

# STEP 2
parse-repair: copy-parser build parse-error-sentences clean
	@echo "Finished..."

# STEP 3
move-orig-parses:
	@cd ../../.. && \
	python -m code.data_preparation.parse.move_orig_parses
	@echo "Orig parses successfully moved..."


# helper targets ##################################################################
copy-parser:
	cp -r ../../lib/parser-collins tmp_parser
	@echo "parser copied successfully"

build:
	@echo "building the parser..."
	cd tmp_parser/code && \
	sed -i.bak 's:define PMAXWORDS 120:define PMAXWORDS 250:' sentence.h && \
	make && \
	cd ..

parse-error-sentences:
	@cd ../../.. && \
	python -m code.data_preparation.parse.parse_error_sentences

clean:
	@if [ -d tmp_parser ] ; \
	then \
		rm -r tmp_parser ; \
		echo "removed tmp_parser folder" ; \
	fi ;

