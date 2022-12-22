
#!/usr/bin/env bash

set -e
set -x

mypy portfolioqtopt
black portfolioqtopt tests --check
isort portfolioqtopt tests --check-only
