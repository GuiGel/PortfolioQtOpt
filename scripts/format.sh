#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place qoptimiza tests --exclude=__init__.py
black qoptimiza tests docs/source/simulation
isort qoptimiza tests
