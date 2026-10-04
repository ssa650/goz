#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi
# Existing environments also need the new Mind Monitor receiver.
if ! .venv/bin/python -c 'import pythonosc' >/dev/null 2>&1; then
  .venv/bin/python -m pip install 'python-osc>=1.9,<2'
fi
exec .venv/bin/python -m backend
