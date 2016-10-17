#!/bin/bash

cd /atlas/u/jkuck/rbpf_target_tracking

PACKAGE_DIR=/atlas/u/jkuck/software
export PATH=$PACKAGE_DIR/anaconda2/bin:$PATH
export LD_LIBRARY_PATH=$PACKAGE_DIR/anaconda2/local:$LD_LIBRARY_PATH
conda create -n anaconda_venv python=2.7.12 anaconda

source activate anaconda_venv

cd /atlas/u/jkuck/RANDOM_TEST_GIT_DIR/rbpf_atlas 

python KITTI_helpers/jdk_helper_evaluate_results.py $RESULTS_DIR $USE_CORRECTED_EVAL

source deactivate
