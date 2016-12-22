#!/bin/bash

cd /atlas/u/jkuck/rbpf_atlas 

PACKAGE_DIR=/atlas/u/jkuck/software
export PATH=$PACKAGE_DIR/anaconda2/bin:$PATH
export LD_LIBRARY_PATH=$PACKAGE_DIR/anaconda2/local:$LD_LIBRARY_PATH

#if running for the first time, uncomment next line or run manually
#conda create -n anaconda_venv python=2.7.12 anaconda

source activate anaconda_venv

#if running for the first time, uncomment next 3 lines or run manually
#conda install -n anaconda_venv numpy
#conda install -n anaconda_venv -c veeresht filterpy=0.1.2
#conda install -n anaconda_venv -c omnia munkres=1.0.7

echo "Using python from:"
which python
echo "------------------------------------------------------------"

echo "pwd:"
pwd

python rbpf_KITTI_det_scores.py $num_particles $include_ignored_gt $include_dontcare_in_gt $use_regionlets $use_mscnn $sort_dets_on_intervals $RUN_IDX $NUM_RUNS $SEQ_IDX

source deactivate

#cd /atlas/u/jkuck/rbpf_target_tracking
#source venv/bin/activate
#python rbpf_KITTI_det_scores.py 100 False False True True 1 10 11 run

#qsub -I -q atlas -l nodes=atlas1.stanford.edu:ppn=1
