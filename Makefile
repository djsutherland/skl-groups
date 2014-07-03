.PHONY: help docs test

help:
	@echo "test - run unit tests"
	@echo "docs - generate Sphinx HTML docs"

docs:
	rm -rf docs/_build/html docs/_build/plot_directive
	$(MAKE) -C docs html

test:
	nosetests
