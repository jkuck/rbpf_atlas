import subprocess
import os
import errno
import sys

NUM_RUNS=3
SEQUENCES_TO_PROCESS = [i for i in range(21)]
#SEQUENCES_TO_PROCESS = [11]
#SEQUENCES_TO_PROCESS = [13]
#NUM_PARTICLES_TO_TEST = [25, 100]
NUM_PARTICLES_TO_TEST = [100]
DIRECTORY_OF_ALL_RESULTS = './results'
#CUR_EXPERIMENT_BATCH_NAME = 'gen_data'

#CUR_EXPERIMENT_BATCH_NAME = 'genData_origRBPF_multMeas_probDet95_clutLambdaPoint1_noise05_noShuffle_beta1'
#CUR_EXPERIMENT_BATCH_NAME = 'genData_origRBPF_multMeas_fixedRounding_resampleRatio4_scaled_ShuffleMeas_timeScaled_PQdiv100'
CUR_EXPERIMENT_BATCH_NAME = 'fixedWeights_predictLSTM_variableR'

def get_description_of_run(include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals,
                           use_regionlets, use_mscnn):
    #use regionlets and mscnn
    if use_regionlets and use_mscnn:
        if (not include_ignored_gt) and (not include_dontcare_in_gt)\
            and sort_dets_on_intervals:
            description_of_run = "mscnn_and_regionlets_with_score_intervals"
        elif (not include_ignored_gt) and (not include_dontcare_in_gt)\
            and (not sort_dets_on_intervals):
            description_of_run = "mscnn_and_regionlets_no_score_intervals"
        else:
            print "Unexpected combination of boolean arguments"
            print include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals, use_regionlets, use_mscnn
            sys.exit(1);

    #use only mscnn
    elif (not use_regionlets) and use_mscnn:
        if (not include_ignored_gt) and (not include_dontcare_in_gt) and \
            sort_dets_on_intervals:
            description_of_run = "mscnn_only_with_score_intervals"

        elif include_ignored_gt and (not include_dontcare_in_gt) and \
            sort_dets_on_intervals:
            description_of_run = "mscnn_only_with_score_intervals_include_ignored_gt"

        elif include_ignored_gt and include_dontcare_in_gt and\
            sort_dets_on_intervals:
            description_of_run = "mscnn_only_with_score_intervals_include_ignored_and_dontcare_in_gt"
        else:
            print "Unexpected combination of boolean arguments"
            sys.exit(1);


    #use only regionlets
    elif use_regionlets and (not use_mscnn):
        if (not include_ignored_gt) and (not include_dontcare_in_gt) and (sort_dets_on_intervals):
                description_of_run = "regionlets_only_with_score_intervals"

        elif (not include_ignored_gt) and (not include_dontcare_in_gt) and (not sort_dets_on_intervals):
            description_of_run = "regionlets_only_no_score_intervals"

        elif (include_ignored_gt) and (not include_dontcare_in_gt) and (sort_dets_on_intervals):
                description_of_run = "regionlets_only_include_ignored_gt_with_score_intervals"
        else:
            print "Unexpected combination of boolean arguments"
            sys.exit(1);


    #error
    else:
        print "Unexpected combination of boolean arguments"
        print use_regionlets, use_mscnn
        sys.exit(1);

    return description_of_run


def run_complete(run_idx, seq_idx, num_particles, include_ignored_gt, include_dontcare_in_gt, 
    sort_dets_on_intervals, use_regionlets, use_mscnn):
    """
    Output:
        - complete: bool, True if this run has been completed already
    """
    description_of_run = get_description_of_run(include_ignored_gt, include_dontcare_in_gt, \
                                                sort_dets_on_intervals, use_regionlets, use_mscnn)
    results_folder_name = '%s/%d_particles' % (description_of_run, num_particles)
    results_folder = '%s/%s/%s' % (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name)
    indicate_run_complete_filename = '%s/results_by_run/run_%d/seq_%d_done.txt' % (results_folder, run_idx, seq_idx)
    complete = os.path.isfile(indicate_run_complete_filename)
    return complete

def setup_results_folder(num_particles, include_ignored_gt, include_dontcare_in_gt, \
                         sort_dets_on_intervals, use_regionlets, use_mscnn):
    description_of_run = get_description_of_run(include_ignored_gt, include_dontcare_in_gt, \
                                                sort_dets_on_intervals, use_regionlets, use_mscnn)
    results_folder_name = '%s/%d_particles' % (description_of_run, num_particles)
    results_folder = '%s/%s/%s' % (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name)

    for cur_run_idx in range(1, NUM_RUNS + 1):
        file_name = '%s/results_by_run/run_%d/%s.txt' % (results_folder, cur_run_idx, 'random_name')
        if not os.path.exists(os.path.dirname(file_name)):
            try:
                os.makedirs(os.path.dirname(file_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

def submit_single_qsub_job(use_regionlets, use_mscnn, num_particles, include_ignored_gt=False, include_dontcare_in_gt=False, 
    sort_dets_on_intervals=True, run_idx=-1, seq_idx=-1):
    if not run_complete(run_idx, seq_idx, num_particles, include_ignored_gt, include_dontcare_in_gt, 
                        sort_dets_on_intervals, use_regionlets, use_mscnn):

        command = 'qsub -q atlas -l nodes=1:ppn=1 -v num_particles=%d,include_ignored_gt=%s,' \
                'include_dontcare_in_gt=%s,use_regionlets=%s,use_mscnn=%s,sort_dets_on_intervals=%s,' \
                'RUN_IDX=%d,NUM_RUNS=%d,SEQ_IDX=%d setup_rbpf_anaconda_venv.sh' \
                 % (num_particles, include_ignored_gt, include_dontcare_in_gt, use_regionlets, use_mscnn, \
                 sort_dets_on_intervals, run_idx, NUM_RUNS, seq_idx)
        os.system(command)



def submit_single_experiment(use_regionlets, use_mscnn, num_particles, include_ignored_gt=False, include_dontcare_in_gt=False, 
    sort_dets_on_intervals=True):
    setup_results_folder(num_particles, include_ignored_gt, include_dontcare_in_gt, \
                         sort_dets_on_intervals, use_regionlets, use_mscnn)
    for run_idx in range(1, NUM_RUNS+1):
        for seq_idx in SEQUENCES_TO_PROCESS:
            submit_single_qsub_job(use_regionlets, use_mscnn, num_particles=num_particles, include_ignored_gt=include_ignored_gt, 
                include_dontcare_in_gt=include_dontcare_in_gt,
                sort_dets_on_intervals=sort_dets_on_intervals, run_idx=run_idx, seq_idx=seq_idx)


if __name__ == "__main__":
    #mscnn_only_with_score_intervals
    for num_particles in NUM_PARTICLES_TO_TEST:
        submit_single_experiment(use_regionlets=False, use_mscnn=True, num_particles=num_particles, 
                                include_ignored_gt=False, include_dontcare_in_gt=False, 
                                sort_dets_on_intervals=True)
    
    #mscnn_and_regionlets_with_score_intervals
    for num_particles in NUM_PARTICLES_TO_TEST:
        submit_single_experiment(use_regionlets=True, use_mscnn=True, num_particles=num_particles, 
                                include_ignored_gt=False, include_dontcare_in_gt=False, 
                                sort_dets_on_intervals=True)

    #regionlets_only_with_score_intervals
    for num_particles in NUM_PARTICLES_TO_TEST:
        submit_single_experiment(use_regionlets=True, use_mscnn=False, num_particles=num_particles, 
                                include_ignored_gt=False, include_dontcare_in_gt=False, 
                                sort_dets_on_intervals=True)
 
#    #regionlets_only_no_score_intervals
#    for num_particles in NUM_PARTICLES_TO_TEST:
#        submit_single_experiment(use_regionlets=True, use_mscnn=False, num_particles=num_particles, 
#                                include_ignored_gt=False, include_dontcare_in_gt=False, 
#                                sort_dets_on_intervals=False)
#

#   #mscnn_only_with_score_intervals_include_ignored_gt
#   for num_particles in NUM_PARTICLES_TO_TEST:
#       submit_single_experiment(use_regionlets=False, use_mscnn=True, num_particles=num_particles, 
#                               include_ignored_gt=True, include_dontcare_in_gt=False, 
#                               sort_dets_on_intervals=True)
#
#   #mscnn_only_with_score_intervals_include_ignored_and_dontcare_in_gt
#   for num_particles in NUM_PARTICLES_TO_TEST:
#       submit_single_experiment(use_regionlets=False, use_mscnn=True, num_particles=num_particles, 
#                               include_ignored_gt=True, include_dontcare_in_gt=True, 
#                               sort_dets_on_intervals=True)
#