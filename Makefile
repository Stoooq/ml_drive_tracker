MODEL ?= pytorch

serve:
	uv run python main.py --serve

detect:
	uv run python main.py --video $(VIDEO) --model $(MODEL)

export:
	uv run python ml/exporter.py

train:
	uv run python ml/trainer.py

benchmark:
	uv run python core/benchmark.py

build-cpp:
	cd cpp && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4

lint:
	uv run ruff check .
	uv run mypy .

docker-up:
	docker-compose up --build

mlflow-ui:
	uv run mlflow ui