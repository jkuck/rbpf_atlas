import os

#DIRECTORY_TO_SEARCH = './results/origRBPF_KITTI_DATA_learnedKFparams'
DIRECTORY_TO_SEARCH = './results/fixedWeights_predictLSTM_DansMedianVarCutoff'


def find_and_eval_results(directory_to_search, seq_idx_to_eval=[i for i in range(21)], info_by_run=None):
    """
    Inputs:
    - seq_idx_to_eval: a list of sequence indices to evaluate NOT USING NOW
    - directory_to_search: string, director name to begin search from (e.g. '/Users/jkuck/rotation3/pykalman')
    """

    for filename in os.listdir(directory_to_search):
        if filename == 'results_by_run':
            print "about to eval: ", directory_to_search

            if (not os.path.isfile(directory_to_search + '/OLD_evaluation_metrics.txt')):
                command = 'qsub -q atlas -l nodes=1:ppn=1 -v RESULTS_DIR=%s,USE_CORRECTED_EVAL=%s \
                           submit_single_eval_job.sh' % (directory_to_search, False)
                os.system(command)

            if (not os.path.isfile(directory_to_search + '/NEW_evaluation_metrics.txt')):
                command = 'qsub -q atlas -l nodes=1:ppn=1 -v RESULTS_DIR=%s,USE_CORRECTED_EVAL=%s \
                           submit_single_eval_job.sh' % (directory_to_search, True)
                os.system(command)

        elif os.path.isdir(directory_to_search + '/' + filename):
            find_and_eval_results(directory_to_search + '/' + filename, seq_idx_to_eval, info_by_run)


if __name__ == "__main__":
    find_and_eval_results(DIRECTORY_TO_SEARCH)

