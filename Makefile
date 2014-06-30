.PHONY: help docs test

help:
	@echo "test - run unit tests"
	@echo "docs - generate Sphinx HTML docs"

docs:
	rm -rf docs/_build/html
	$(MAKE) -C docs html

test:
	nosetests
