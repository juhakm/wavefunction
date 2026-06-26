# Automatically extract project name from pyproject.toml
PROJECT := $(shell grep -m 1 '^name = ' pyproject.toml | awk -F'"' '{print $$2}')

.PHONY: help clean package check upload install uninstall unittest pyright coverage mypy

help:
	@echo "Available commands:"
	@echo "  make package     - Build source and wheel distributions"
	@echo "  make check       - Verify build distribution compatibility with PyPI"
	@echo "  make upload      - Build, verify, and upload package to PyPI"
	@echo "  make install     - Install package locally in editable mode"
	@echo "  make uninstall   - Uninstall local package"
	@echo "  make unittest    - Run pytest suite"
	@echo "  make coverage    - Run tests with XML coverage reporting"
	@echo "  make pyright     - Run pyright static type analysis"
	@echo "  make mypy        - Run mypy static type analysis"
	@echo "  make clean       - Remove built distribution files"

package: clean
	@python3 -m build

check:
	@python3 -m twine check dist/*

# Chaining these ensures you always upload a fresh, verified build
upload: package check
	@python3 -m twine upload --repository pypi dist/* --verbose

install:
	@python3 -m pip install -e .

uninstall:
	@python3 -m pip uninstall -y $(PROJECT)

clean:
	@rm -rf dist/ build/ *.egg-info

unittest:
	@python3 -m pytest

pyright:
	pyright

coverage:
	python3 -m pytest --cov=$(PROJECT) --cov-report=xml

mypy:
	python3 -m mypy --show-traceback .
