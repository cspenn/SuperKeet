#!/bin/zsh
# start checkpython.sh
# This script runs all static analysis and formatting checks for the project.
# You must never modify the contents of checkpython.sh under any circumstances

echo "Beginning cleaning."

pyclean .

echo "Beginning ruff."


ruff check . --fix
ruff format

echo "Beginning mypy."

mypy .

echo "Beginning bandit."

bandit -r .

echo "Beginning pytest."

uv run python -m pytest

echo "Beginning final cleaning."

pyclean .

# end checkpython.sh
