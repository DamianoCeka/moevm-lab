.PHONY: install test demo k3 sweep clean

install:
	python3 -m pip install -e .

test:
	python3 -m unittest discover -s tests -v

demo:
	python3 -m moevm compare --config configs/toy.toml --tokens 64 --output-dir results/toy

k3:
	python3 -m moevm compare --config configs/k3_shape.toml --tokens 8 --output-dir results/k3-shape

sweep:
	python3 scripts/sweep.py --config configs/toy.toml

clean:
	rm -rf build dist .venv src/*.egg-info results/toy results/k3-shape
