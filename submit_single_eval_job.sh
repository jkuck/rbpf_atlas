#!/bin/bash

cd /atlas/u/jkuck/rbpf_atlas


PACKAGE_DIR=/atlas/u/jkuck/software
export PATH=$PACKAGE_DIR/anaconda2/bin:$PATH
export LD_LIBRARY_PATH=$PACKAGE_DIR/anaconda2/local:$LD_LIBRARY_PATH
conda create -n anaconda_venv python=2.7.12 anaconda

source activate anaconda_venv

python KITTI_helpers/jdk_helper_evaluate_results.py $RESULTS_DIR $USE_CORRECTED_EVAL

source deactivate
