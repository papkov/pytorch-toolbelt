.PHONY: clean build publish format lint

MODULES_PATH=./pytorch_toolbelt
TEST_PATH=./tests
BLACK_CFG=./black.toml


build: clean
	python -m pip install --upgrade --quiet setuptools wheel twine
	python setup.py --quiet sdist bdist_wheel

publish: build
	python -m twine check dist/*
	python -m twine upload dist/*

clean:
	rm -r build dist *.egg-info || true

format:
	isort $(MODULES_PATH) $(TEST_PATH)
	black --config $(BLACK_CFG) $(MODULES_PATH) $(TEST_PATH)

lint:
	isort -c $(MODULES_PATH) $(TEST_PATH)
	black --config $(BLACK_CFG) --check $(MODULES_PATH) $(TEST_PATH)
	mypy $(MODULES_PATH) $(TEST_PATH)