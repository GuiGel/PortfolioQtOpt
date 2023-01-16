#!/bin/sh -e
set -x

cd docs
make clean
make html 
cd ./build/html
python -m http.server
