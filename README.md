# rbpf_atlas
##1. Log into Atlas
##2. Create a directory for yourself and go to it:
$ cd /atlas/u
$ mkdir <name of your choice>      #for intance mkdir <your_id>
$ cd <your_id>

##3. Clone this repository:
$ git clone https://github.com/jkuck/rbpf_atlas
$ cd rbpf_atlas/

##4. Now you should be in the directory /atlas/u/<your_id>/rbpf_atlas.
   Edit line 12 of submit_single_eval_job.sh and line 17 of setup_rbpf_anaconda_venv.sh from:
cd /atlas/u/jkuck/RANDOM_TEST_GIT_DIR/rbpf_atlas
   to instead read:
cd /atlas/u/<your_id>/rbpf_atlas

##5. Run an experiment:
$ python run_experiment_batch.py

##6. Evaluate the experiment's results:
$ python find_and_evaluate_results.py
