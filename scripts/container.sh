#!/usr/bin/env bash

set -e
set -x

docker build -t qoptimiza .
docker run -p 8501:8501 qoptimiza