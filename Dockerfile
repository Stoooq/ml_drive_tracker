FROM arm64v8/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    git \
    make \
    python3 \
    python3-pip \
    libopencv-dev

WORKDIR /app

# Python deps cached separately — reinstalled only when pyproject.toml or uv.lock changes
COPY pyproject.toml uv.lock .python-version ./
RUN pip3 install uv && uv sync --no-install-project

# C++ dependency download cached separately — TFLite re-downloaded only when CMakeLists.txt changes
COPY cpp/CMakeLists.txt cpp/CMakeLists.txt
RUN mkdir -p cpp/build && \
    touch cpp/main.cpp cpp/detector.cpp cpp/detector.hpp cpp/detector_lib.cpp && \
    cd cpp/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release

# C++ compilation — runs on every change to .cpp/.hpp files
COPY cpp/ cpp/
RUN cd cpp/build && cmake --build . --parallel $(nproc)

# Remaining Python source — copied last so changes here don't invalidate C++ cache
COPY . .

ENV CPP_TFLITE_DETECT_PATH=cpp/build/libtflite_detect_lib.so

ENTRYPOINT ["uv", "run", "main.py"]
