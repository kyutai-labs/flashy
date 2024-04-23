default: linter tests

install:
	pip install -U pip
	pip install numpy  # installing numpy to deactivate stupid warnings for pytorch even when not used.
	pip install -U git+https://github.com/facebookincubator/submitit@main#egg=submitit
	pip install -U git+https://git@github.com/facebookresearch/dora#egg=dora-search
	pip install -U -e '.[dev]'


install_ci:
	pip install -U pip
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install --no-use-pep517 git+https://github.com/facebookincubator/submitit@main#egg=submitit
	pip install --no-use-pep517 git+https://git@github.com/facebookresearch/dora#egg=dora-search
	pip install --no-use-pep517 -e '.[dev]'


linter:
	flake8 flashy && mypy flashy

docs:
	pdoc3 --html -o docs -f flashy

tests:
	coverage run -m pytest tests
	coverage report --include 'flashy/*'

upload: docs
	rsync -ar docs bob:www/share/flashy/

dist:
	python setup.py sdist

.PHONY: linter tests docs upload dist
