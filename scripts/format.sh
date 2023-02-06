#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place portfolioqtopt tests application --exclude=__init__.py
black portfolioqtopt tests docs/source/simulation application
isort portfolioqtopt tests application
