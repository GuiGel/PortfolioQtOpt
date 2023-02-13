
#!/usr/bin/env bash

set -e
set -x

mypy qoptimiza
black qoptimiza tests --check
isort qoptimiza tests --check-only
