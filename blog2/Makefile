.PHONY: help book clean serve

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install     to install the necessary dependencies for jupyter-book to build"
	@echo "  book        to convert the `content/` folder into Jekyll markdown in `_build/`"
	@echo "  clean       to clean out site build files"
	@echo "  runall      to run all notebooks in-place, capturing outputs with the notebook"
	@echo "  serve       to serve the repository locally with Jekyll"
	@echo "  html        to build the site HTML locally with Jekyll and store in _site/"
	@echo "  publish     to build the site HTML locally with Jekyll and store in _site/"


clean:
	rm -rf _build
	rm -rf blog/blog2

install:
	gem install bundler
	bundle install

book:
	python scripts/make_toc.py
	jupyter-book build ./

html:
	bundle exec jekyll build
	touch ../blog/blog2/.nojekyll

publish:
	cd blog && sh commit.sh

runall:
	jupyter-book run ./content

serve:
	bundle exec guard
