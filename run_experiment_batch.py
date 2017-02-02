import subprocess
import os
import errno
import sys

NUM_RUNS=1
SEQUENCES_TO_PROCESS = [i for i in range(21)]
#SEQUENCES_TO_PROCESS = [0]
#SEQUENCES_TO_PROCESS = [11]
#SEQUENCES_TO_PROCESS = [13]
#NUM_PARTICLES_TO_TEST = [25, 100]
NUM_PARTICLES_TO_TEST = [100]
#DIRECTORY_OF_ALL_RESULTS = './ICML_prep_correctedOnline/propose_k=1_nearest_targets'
DIRECTORY_OF_ALL_RESULTS = './ICML_prep_debug2'

#CUR_EXPERIMENT_BATCH_NAME = 'genData_origRBPF_multMeas_probDet95_clutLambdaPoint1_noise05_noShuffle_beta1'
#CUR_EXPERIMENT_BATCH_NAME = 'genData_origRBPF_multMeas_fixedRounding_resampleRatio4_scaled_ShuffleMeas_timeScaled_PQdiv100'


#CUR_EXPERIMENT_BATCH_NAME = 'Rto0_4xQ_multMeas1update_online3frameDelay2'
CUR_EXPERIMENT_BATCH_NAME = 'CHECK_1_NEAREST_TARGETS/Rto0_4xQ_multMeas1update_online3frameDelay'
#CUR_EXPERIMENT_BATCH_NAME = 'CHECK_K_NEAREST_TARGETS=False/Reference/Rto0_4xQ_max1MeasUpdate_online3frameDelay'
#CUR_EXPERIMENT_BATCH_NAME = '/Reference/Rto0_4xQ_max1MeasUpdate_online0frameDelay'
#CUR_EXPERIMENT_BATCH_NAME = 'measuredR_1xQ_max1MeasUpdate_online3frameDelay'

def get_description_of_run(include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals,
                           use_regionlets, det1_name, det2_name):

    if det2_name == 'None' or det2_name == None:
        if (not include_ignored_gt) and (not include_dontcare_in_gt)\
            and sort_dets_on_intervals:
            description_of_run = "%s_with_score_intervals" % (det1_name)
        elif (not include_ignored_gt) and (not include_dontcare_in_gt)\
            and (not sort_dets_on_intervals):
            description_of_run = "%s_no_score_intervals" % (det1_name)
        else:
            print "Unexpected combination of boolean arguments"
            print include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals, use_regionlets, use_mscnn
            sys.exit(1);

    else:

        if (not include_ignored_gt) and (not include_dontcare_in_gt)\
            and sort_dets_on_intervals:
            description_of_run = "%s_%s_with_score_intervals" % (det1_name, det2_name)
        elif (not include_ignored_gt) and (not include_dontcare_in_gt)\
            and (not sort_dets_on_intervals):
            description_of_run = "%s_%s_no_score_intervals" % (det1_name, det2_name)
        else:
            print "Unexpected combination of boolean arguments"
            print include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals, use_regionlets, use_mscnn
            sys.exit(1);


#    #use regionlets and mscnn
#    if use_regionlets and use_mscnn:
#        if (not include_ignored_gt) and (not include_dontcare_in_gt)\
#            and sort_dets_on_intervals:
#            description_of_run = "mscnn_and_regionlets_with_score_intervals"
#        elif (not include_ignored_gt) and (not include_dontcare_in_gt)\
#            and (not sort_dets_on_intervals):
#            description_of_run = "mscnn_and_regionlets_no_score_intervals"
#        else:
#            print "Unexpected combination of boolean arguments"
#            print include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals, use_regionlets, use_mscnn
#            sys.exit(1);
#
#    #use only mscnn
#    elif (not use_regionlets) and use_mscnn:
#        if (not include_ignored_gt) and (not include_dontcare_in_gt) and \
#            sort_dets_on_intervals:
#            description_of_run = "mscnn_only_with_score_intervals"
#
#        elif (not include_ignored_gt) and (not include_dontcare_in_gt) and \
#            (not sort_dets_on_intervals):
#            description_of_run = "mscnn_only_no_score_intervals"
#
#        elif include_ignored_gt and (not include_dontcare_in_gt) and \
#            sort_dets_on_intervals:
#            description_of_run = "mscnn_only_with_score_intervals_include_ignored_gt"
#
#        elif include_ignored_gt and include_dontcare_in_gt and\
#            sort_dets_on_intervals:
#            description_of_run = "mscnn_only_with_score_intervals_include_ignored_and_dontcare_in_gt"
#        else:
#            print "Unexpected combination of boolean arguments"
#            sys.exit(1);
#
#
#    #use only regionlets
#    elif use_regionlets and (not use_mscnn):
#        if (not include_ignored_gt) and (not include_dontcare_in_gt) and (sort_dets_on_intervals):
#                description_of_run = "regionlets_only_with_score_intervals"
#
#        elif (not include_ignored_gt) and (not include_dontcare_in_gt) and (not sort_dets_on_intervals):
#            description_of_run = "regionlets_only_no_score_intervals"
#
#        elif (include_ignored_gt) and (not include_dontcare_in_gt) and (sort_dets_on_intervals):
#                description_of_run = "regionlets_only_include_ignored_gt_with_score_intervals"
#        else:
#            print "Unexpected combination of boolean arguments"
#            sys.exit(1);
#
#
#    #error
#    else:
#        print "Unexpected combination of boolean arguments"
#        print use_regionlets, use_mscnn
#        sys.exit(1);
#
    return description_of_run


def run_complete(run_idx, seq_idx, num_particles, include_ignored_gt, include_dontcare_in_gt, 
    sort_dets_on_intervals, use_regionlets, det1_name, det2_name):
    """
    Output:
        - complete: bool, True if this run has been completed already
    """
    description_of_run = get_description_of_run(include_ignored_gt, include_dontcare_in_gt, \
                                                sort_dets_on_intervals, use_regionlets, det1_name, det2_name)
    results_folder_name = '%s/%d_particles' % (description_of_run, num_particles)
    results_folder = '%s/%s/%s' % (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name)
    indicate_run_complete_filename = '%s/results_by_run/run_%d/seq_%d_done.txt' % (results_folder, run_idx, seq_idx)
    complete = os.path.isfile(indicate_run_complete_filename)
    return complete

def setup_results_folder(num_particles, include_ignored_gt, include_dontcare_in_gt, \
                         sort_dets_on_intervals, use_regionlets, det1_name, det2_name):
    description_of_run = get_description_of_run(include_ignored_gt, include_dontcare_in_gt, \
                                                sort_dets_on_intervals, use_regionlets, det1_name, det2_name)
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

def submit_single_qsub_job(use_regionlets, det1_name, det2_name, num_particles, include_ignored_gt=False, include_dontcare_in_gt=False, 
    sort_dets_on_intervals=True, run_idx=-1, seq_idx=-1):
    if not run_complete(run_idx, seq_idx, num_particles, include_ignored_gt, include_dontcare_in_gt, 
                        sort_dets_on_intervals, use_regionlets, det1_name, det2_name):

        command = 'qsub -q atlas -l nodes=1:ppn=1 -v num_particles=%d,include_ignored_gt=%s,' \
                'include_dontcare_in_gt=%s,use_regionlets=%s,det1_name=%s,det2_name=%s,sort_dets_on_intervals=%s,' \
                'RUN_IDX=%d,NUM_RUNS=%d,SEQ_IDX=%d setup_rbpf_anaconda_venv.sh' \
                 % (num_particles, include_ignored_gt, include_dontcare_in_gt, use_regionlets, det1_name, det2_name, \
                 sort_dets_on_intervals, run_idx, NUM_RUNS, seq_idx)
        os.system(command)



def submit_single_experiment(use_regionlets, det1_name, det2_name, num_particles, include_ignored_gt=False, include_dontcare_in_gt=False, 
    sort_dets_on_intervals=True):
    setup_results_folder(num_particles, include_ignored_gt, include_dontcare_in_gt, \
                         sort_dets_on_intervals, use_regionlets, det1_name, det2_name)
    for run_idx in range(1, NUM_RUNS+1):
        for seq_idx in SEQUENCES_TO_PROCESS:
            submit_single_qsub_job(use_regionlets, det1_name, det2_name, num_particles=num_particles, include_ignored_gt=include_ignored_gt, 
                include_dontcare_in_gt=include_dontcare_in_gt,
                sort_dets_on_intervals=sort_dets_on_intervals, run_idx=run_idx, seq_idx=seq_idx)


if __name__ == "__main__":

#    for num_particles in NUM_PARTICLES_TO_TEST:
#            submit_single_experiment(use_regionlets=False, det1_name = 'mscnn', det2_name = 'regionlets', num_particles=num_particles, 
#                                include_ignored_gt=False, include_dontcare_in_gt=False, 
#                                sort_dets_on_intervals=True)


    for num_particles in NUM_PARTICLES_TO_TEST:
        submit_single_experiment(use_regionlets=False, det1_name = 'mscnn', det2_name = 'regionlets', num_particles=num_particles, 
                            include_ignored_gt=False, include_dontcare_in_gt=False, 
                            sort_dets_on_intervals=True)

####    for num_particles in NUM_PARTICLES_TO_TEST:
#####        for det1_name in ['3dop', 'mono3d', 'mv3d', 'mscnn', 'regionlets']:
####        for det1_name in ['3dop', 'mono3d', 'mv3d']:
####            submit_single_experiment(use_regionlets=False, det1_name = det1_name, det2_name = 'None', num_particles=num_particles, 
####                                include_ignored_gt=False, include_dontcare_in_gt=False, 
####                                sort_dets_on_intervals=True)
####            submit_single_experiment(use_regionlets=False, det1_name = det1_name, det2_name = 'None', num_particles=num_particles, 
####                                include_ignored_gt=False, include_dontcare_in_gt=False, 
####                                sort_dets_on_intervals=False)

#    for num_particles in NUM_PARTICLES_TO_TEST:
#        for det2_name in ['3dop', 'mono3d', 'mv3d']:
#            submit_single_experiment(use_regionlets=False, det1_name = 'mscnn', det2_name = det2_name, num_particles=num_particles, 
#                                include_ignored_gt=False, include_dontcare_in_gt=False, 
#                                sort_dets_on_intervals=True)
#            submit_single_experiment(use_regionlets=False, det1_name = 'mscnn', det2_name = det2_name, num_particles=num_particles, 
#                                include_ignored_gt=False, include_dontcare_in_gt=False, 
#                                sort_dets_on_intervals=False)

#    #mscnn_only_with_score_intervals
#    for num_particles in NUM_PARTICLES_TO_TEST:
#        submit_single_experiment(use_regionlets=False, use_mscnn=True, num_particles=num_particles, 
#                                include_ignored_gt=False, include_dontcare_in_gt=False, 
#                                sort_dets_on_intervals=True)
#    
#    #mscnn_and_regionlets_with_score_intervals
#    for num_particles in NUM_PARTICLES_TO_TEST:
#        submit_single_experiment(use_regionlets=True, use_mscnn=True, num_particles=num_particles, 
#                                include_ignored_gt=False, include_dontcare_in_gt=False, 
#                                sort_dets_on_intervals=True)
#
#    #regionlets_only_with_score_intervals
#    for num_particles in NUM_PARTICLES_TO_TEST:
#        submit_single_experiment(use_regionlets=True, use_mscnn=False, num_particles=num_particles, 
#                                include_ignored_gt=False, include_dontcare_in_gt=False, 
#                                sort_dets_on_intervals=True)

#    #mscnn_only_no_score_intervals
#    for num_particles in NUM_PARTICLES_TO_TEST:
#        submit_single_experiment(use_regionlets=False, use_mscnn=True, num_particles=num_particles, 
#                                include_ignored_gt=False, include_dontcare_in_gt=False, 
#                                sort_dets_on_intervals=False)
#    
#    #mscnn_and_regionlets_no_score_intervals
#    for num_particles in NUM_PARTICLES_TO_TEST:
#        submit_single_experiment(use_regionlets=True, use_mscnn=True, num_particles=num_particles, 
#                                include_ignored_gt=False, include_dontcare_in_gt=False, 
#                                sort_dets_on_intervals=False)
#
#    #regionlets_only_no_score_intervals
#    for num_particles in NUM_PARTICLES_TO_TEST:
#        submit_single_experiment(use_regionlets=True, use_mscnn=False, num_particles=num_particles, 
#                                include_ignored_gt=False, include_dontcare_in_gt=False, 
#                                sort_dets_on_intervals=False)


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