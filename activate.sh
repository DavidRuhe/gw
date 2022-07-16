#!/bin/sh

ENV_DIR='.venv'

if [[ ! -d $ENV_DIR ]]; then
    python3 -m venv $ENV_DIR
fi

source $ENV_DIR/bin/activate
export DATAROOT=$(pwd)/datasets/
cd src/
export PYTHONPATH=$(pwd)
