.PHONY: submodules build-betti dev lint clean

submodules:
	./scripts/setup_submodules.sh

build-betti: submodules
	./scripts/build_betti.sh

dev:                      ## install this package in editable mode
	pip install -e .

test:
	pytest -q

clean:
	rm -rf build dist .tox **/__pycache__
