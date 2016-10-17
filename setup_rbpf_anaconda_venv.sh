#!/bin/bash

cd /atlas/u/jkuck/RANDOM_TEST_GIT_DIR/rbpf_atlas 

PACKAGE_DIR=/atlas/u/jkuck/software
export PATH=$PACKAGE_DIR/anaconda2/bin:$PATH
export LD_LIBRARY_PATH=$PACKAGE_DIR/anaconda2/local:$LD_LIBRARY_PATH

#already created, don't run again
#conda create -n anaconda_venv python=2.7.12 anaconda


source activate anaconda_venv
#conda install -n anaconda_venv numpy
#conda install -n anaconda_venv filterpy
#conda install -n anaconda_venv munkres

echo "Using python from:"
which python
echo "------------------------------------------------------------"

python rbpf_KITTI_det_scores.py $num_particles $include_ignored_gt $include_dontcare_in_gt $use_regionlets_and_lsvm $sort_dets_on_intervals $RUN_IDX $NUM_RUNS $SEQ_IDX $PERIPHERAL

source deactivate

#cd /atlas/u/jkuck/rbpf_target_tracking
#source venv/bin/activate
#python rbpf_KITTI_det_scores.py 100 False False True True 1 10 11 run

#qsub -I -q atlas -l nodes=atlas1.stanford.edu:ppn=1
