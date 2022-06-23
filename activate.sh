#!/bin/sh

ENV_DIR='.venv'

if [[ ! -d $ENV_DIR ]]; then
    python3 -m venv $ENV_DIR
fi

source $ENV_DIR/bin/activate
cd src/
export PYTHONPATH=$(pwd)
