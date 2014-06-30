.PHONY: help docs test

help:
	@echo "test - run unit tests"
	@echo "docs - generate Sphinx HTML docs"

docs:
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

test:
	nosetests
