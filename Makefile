
all: lint lint_markdown

POETRY_NO_ROOT:= --no-root

dev_setup:
	poetry install $(POETRY_NO_ROOT) $(POETRY_OPTION)

setup:
	poetry install $(POETRY_OPTION)

TARGETS:=scripts

flake8:
	find $(TARGETS) | grep '\.py$$' | xargs flake8
autopep8:
	find $(TARGETS) | grep '\.py$$' | xargs autopep8 -d | diff /dev/null -
mypy:
	find $(TARGETS) | grep '\.py$$' | xargs mypy --python-version 3.7 --check-untyped-defs --strict-equality --no-implicit-optional
isort:
	find $(TARGETS) | grep '\.py$$' | xargs isort --diff | diff /dev/null -
pydocstyle:
	find $(TARGETS) | grep '\.py$$' | grep -v run_clm.py | \
		xargs pydocstyle --ignore=D100,D101,D102,D103,D104,D105,D107,D203,D212,D400,D415

yamllint:
	find . -name '*.yml' -type f | xargs yamllint --no-warnings

check_firstline:
	find $(TARGETS) ./.circleci -type f | grep -v -e 'git' -e 'idea' -e 'mypy' -e 'python_env' | grep -e 'py$$' | grep -v '__init__' | grep -v third | xargs python3 .circleci/check_head.py


lint: flake8 autopep8 mypy isort yamllint check_firstline pydocstyle

_run_isort:
	isort -rc .

setup_node_module:
	npm install markdownlint-cli

lint_markdown:
	find . -type d -o -type f -name '*.md' -print \
	| grep -v \.venv \
	| grep -v node_modules \
	| xargs npx markdownlint --config ./.markdownlint.json

.PHONY: all setup \
	flake8 autopep8 mypy isort yamllint\
	check_firstline \
	lint \
	_run_isort \
	setup_node_module lint_markdown circleci_local

.DELETE_ON_ERROR:

circleci_local:
	circleci local execute
