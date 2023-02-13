#!/usr/bin/env bash

set -e
set -x

poetry export --without-hashes --format=requirements.txt > requirements.txt