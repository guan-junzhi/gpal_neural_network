#!/bin/bash

pwd
echo $0
test_path=$(pwd)
PYTHONPATH="${test_path}:$PYTHONPATH"
echo PYTHONPATH=${PYTHONPATH}
export PYTHONPATH
