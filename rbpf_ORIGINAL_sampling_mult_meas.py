import numpy as np
from numpy.linalg import inv
import random

import math
import scipy.stats as ss
from scipy.special import gdtrc

from gen_data import SCALED
from gen_data import NOISE_SD
from gen_data import default_time_step
from gen_data import prob_detection

class Parameters:
    def __init__(self, target_emission_probs, clutter_probabilities,\
                 birth_probabilities, meas_noise_cov, R_default, H,\
                 USE_PYTHON_GAUSSIAN, USE_CONSTANT_R, score_intervals,\
                 p_birth_likelihood, p_clutter_likelihood):
        '''
        Inputs:
        -  score_intervals: list of lists, where score_intervals[i] is a list
            specifying the score intervals for measurement source i.  
            score_intervals[i][j] specifies the lower bound for the jth score
            interval corresponding to measurement source i (0 indexed).
        '''
        self.target_emission_probs = target_emission_probs
        self.clutter_probabilities = clutter_probabilities
        self.birth_probabilities = birth_probabilities

        self.meas_noise_cov = None
        self.R_default = R_default
#        if SCALED:
#            self.R_default = np.array([[ (NOISE_SD*300)**2,             0.0],
#                                       [          0.0,   (NOISE_SD*90)**2]])
#        else:
#            self.R_default = np.array([[ (NOISE_SD)**2,         0.0],
#                                       [      0.0,   (NOISE_SD)**2]])

        self.H = H

        self.USE_PYTHON_GAUSSIAN = USE_PYTHON_GAUSSIAN
        self.USE_CONSTANT_R = True

        self.score_intervals = score_intervals

        self.p_birth_likelihood = p_birth_likelihood 
        self.p_clutter_likelihood = p_clutter_likelihood


    def get_score_index(self, score, meas_source_index):
        """
        Inputs:
        - score: the score of a detection

        Output:
        - index: output the 0 indexed score interval this score falls into
        """

        index = 0
        for i in range(1, len(self.score_intervals[meas_source_index])):
            if(score > self.score_intervals[meas_source_index][i]):
                index += 1
            else:
                break
        print len(self.score_intervals[meas_source_index])
        print self.score_intervals[meas_source_index]
        print index
        assert(score > self.score_intervals[meas_source_index][index]), (score, self.score_intervals[meas_source_index][index], self.score_intervals[meas_source_index][index+1]) 
        if(index < len(self.score_intervals[meas_source_index]) - 1):
            assert(score <= self.score_intervals[meas_source_index][index+1]), (score, self.score_intervals[meas_source_index][index], self.score_intervals[meas_source_index][index+1])
        return index

    def emission_prior(self, meas_source_index, meas_score):
        score_index = self.get_score_index(meas_score, meas_source_index)
        return self.target_emission_probs[meas_source_index][score_index]

    def clutter_prior(self, meas_source_index, meas_score, clutter_count):
        '''
        The prior probability of clutter_count number of clutter measurements with score 
        given by meas_score from the measurement source with index meas_source_index
        '''    
        score_index = self.get_score_index(meas_score, meas_source_index)    
        return self.clutter_probabilities[meas_source_index][score_index][clutter_count]

    def max_clutter_count(self, meas_source_index, meas_score):
        '''
        The maximum clutter count from the specified measurement source and score
        range that has a non-zero prior.
        '''
        score_index = self.get_score_index(meas_score, meas_source_index)    
        return len(self.clutter_probabilities[meas_source_index][score_index]) - 1


    def birth_prior(self, meas_source_index, meas_score, birth_count):
        '''
        The prior probability of birth_count number of births with score given by
        meas_score from the measurement source with index meas_source_index
        '''
        score_index = self.get_score_index(meas_score, meas_source_index)    
        return self.clutter_probabilities[meas_source_index][score_index][birth_count]


    def max_birth_count(self, meas_source_index, meas_score):
        '''
        The maximum birth count from the specified measurement source and score
        range that has a non-zero prior.
        '''
        score_index = self.get_score_index(meas_score, meas_source_index)    
        return len(self.birth_probabilities[meas_source_index][score_index]) - 1

    def check_counts(self, clutter_counts_by_score, birth_counts_by_score, meas_source_index):
        assert(len(clutter_counts_by_score) == len(birth_counts_by_score))
        assert(len(clutter_counts_by_score) == len(self.clutter_probabilities[meas_source_index]))

        for i in range(len(clutter_counts_by_score)):
            assert(0 <= clutter_counts_by_score[i] and clutter_counts_by_score[i] <= len(self.clutter_probabilities[meas_source_index][i]) - 1)
            assert(0 <= birth_counts_by_score[i] and birth_counts_by_score[i] <= len(self.birth_probabilities[meas_source_index][i]) - 1), (birth_counts_by_score[i], len(self.birth_probabilities[meas_source_index][i]) - 1, self.birth_probabilities[meas_source_index][i])


    def get_R(self, meas_source_index, meas_score):
        if self.USE_CONSTANT_R:
            return self.R_default
        else:
            score_index = self.get_score_index(meas_score, meas_source_index)    
            return self.meas_noise_cov[meas_source_index][score_index]

def sample_and_reweight(particle, measurement_lists, \
    cur_time, measurement_scores, params):
    """
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle
    - measurement_lists: a list where measurement_lists[i] is a list of all measurements from the current
        time instance from the ith measurement source (i.e. different object detection algorithms
        or different sensors)
    - measurement_scores: a list where measurement_scores[i] is a list containing scores for every measurement in
        measurement_list[i]
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Output:
    - measurement_associations: A list where measurement_associations[i] is a list of association values
        for each measurements in measurement_lists[i].  Association values correspond to:
        measurement_associations[i][j] = -1 -> measurement is clutter
        measurement_associations[i][j] = particle.targets.living_count -> measurement is a new target
        measurement_associations[i][j] in range [0, particle.targets.living_count-1] -> measurement is of
            particle.targets.living_targets[measurement_associations[i][j]]

    - imprt_re_weight: After processing this measurement the particle's
        importance weight will be:
        new_importance_weight = old_importance_weight * imprt_re_weight
    - targets_to_kill: a list containing the indices of targets that should be killed, beginning
        with the smallest index in increasing order, e.g. [0, 4, 6, 33]
    """

    #get death probabilities for each target in a numpy array
    num_targs = particle.targets.living_count
    p_target_deaths = []
    for target in particle.targets.living_targets:
        p_target_deaths.append(target.death_prob)
        assert(p_target_deaths[len(p_target_deaths) - 1] >= 0 and p_target_deaths[len(p_target_deaths) - 1] <= 1), (p_target_deaths[len(p_target_deaths) - 1], cur_time)


    (targets_to_kill, measurement_associations, importance_reweight) = \
        sample_meas_assoc_and_death(particle, measurement_lists, particle.targets.living_count, p_target_deaths, \
                                    cur_time, measurement_scores, params)



    living_target_indices = []
    for i in range(particle.targets.living_count):
        if(not i in targets_to_kill):
            living_target_indices.append(i)


    assert(num_targs == particle.targets.living_count)
    #sort targets_to_kill
    targets_to_kill.sort()
    #double check targets_to_kill is sorted
    assert(all([targets_to_kill[i] <= targets_to_kill[i+1] for i in xrange(len(targets_to_kill)-1)]))

    return (measurement_associations, targets_to_kill, importance_reweight)

def sample_meas_assoc_and_death(particle, measurement_lists, total_target_count, 
                           p_target_deaths, cur_time, measurement_scores, params):
    """
    Try sampling associations with each measurement sequentially
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle
    - measurement_lists: type list, measurement_lists[i] is a list of all measurements from the current
        time instance from the ith measurement source (i.e. different object detection algorithms
        or different sensors)
    - measurement_scores: type list, measurement_scores[i] is a list containing scores for every measurement in
        measurement_list[i]
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Output:
    - targets_to_kill: a list of targets that have been sampled to die (not killed yet)
    - measurement_associations: type list, measurement_associations[i] is a list of associations for  
        the measurements in measurement_lists[i]
        
    """
    assert(len(measurement_lists) == len(measurement_scores))
    measurement_associations = []
    for meas_source_index in range(len(measurement_lists)):
        (cur_associations, targets_to_kill, importance_reweight) = associate_measurements_sequentially\
            (particle, meas_source_index, measurement_lists[meas_source_index], \
             total_target_count, p_target_deaths, measurement_scores[meas_source_index],\
             params)
        measurement_associations.append(cur_associations)

    assert(len(measurement_associations) == len(measurement_lists))

    #debug
    for meas_source_index in range(len(measurement_associations)):
        for i in range(total_target_count):
            assert(measurement_associations[meas_source_index].count(i) == 0 or \
                   measurement_associations[meas_source_index].count(i) == 1), (measurement_associations[meas_source_index],  measurement_list, total_target_count, p_target_deaths)
    #done debug

    return (targets_to_kill, measurement_associations, importance_reweight)



class SampledEvent:
    def __init__(self, assoc_index, dead_targ_idx):
        self.assoc_index = assoc_index
        self.dead_targ_idx = dead_targ_idx

def associate_measurements_sequentially(particle, meas_source_index, measurement_list, total_target_count, \
    p_target_deaths, measurement_scores, params):

    """
    Try sampling associations with each measurement sequentially
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - measurement_list: a list of all measurements from the current time instance
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Output:
    - list_of_measurement_associations: list of associations for each measurement
    - indices_to_kill: list of target indices to kill
        
    """


#    def bin_pdf(x, p, n):
#        """
#        Probability density function of a Binomial distribution.
#
#        Input:
#          x   - Location where to evaluate the PDF.
#          p - 'Probability' parameter 
#          n   - 'Sample size' parameter
#        """
#        hh = ss.binom(n, p)
#        return hh.pmf(x)    
    if(particle.max_importance_weight):
        print "number of measurements:", len(measurement_list)


    def nCr(n,r):
        return math.factorial(n) / math.factorial(r) / math.factorial(n-r)

    def bin_pdf(x,p,n):
        if (x > n) or (n < 0):
            return 0
        else:
            return nCr(n,x) * p**x * (1-p)**(n-x)


    def death_prob_of_assoc_target(dt):
        #return the probability of death over the next time step,
        #where dt is the duration of one time step
        alpha_death = 2.0
        beta_death = 1.0
        theta_death = 1.0/beta_death
        return 1-gdtrc(theta_death, alpha_death, dt)

    pb = .01 #birth prior
    birth_likelihood = 0.0151
#    birth_likelihood = 1.0/16.0 
#    CP = 0.0 #clutter prior, DOESN"T HAVE ONE!!
    CD = 1.0/16.0 #clutter likelihood
    if SCALED:
        birth_likelihood = birth_likelihood/(300*90)
        CD = CD/(300*90)

    list_of_measurement_associations = []
    indices_to_kill = []

    #sample measurement associations
    birth_count = 0
    clutter_count = 0
    remaining_meas_count = len(measurement_list)

    importance_reweight = 1.0

    #the number of targets that we haven't associated with a measurement or killed
    unas_living_targ_count = particle.targets.living_count
    #the number of targets that we haven't killed
    cur_living_targ_count = particle.targets.living_count
    assert(unas_living_targ_count == len(particle.targets.living_targets))
    for (index, cur_meas) in enumerate(measurement_list):
        meas_score = measurement_scores[index]
        #create proposal distribution for the current measurement with the following entries:
        # 1: clutter, no deaths
        # (unassociated, living target count): target association, no deaths
        # 1: target birth, no deaths
        #
        # (living target count): clutter association, 1 death
        # (unassociated, living target count)*(living target count - 1): target association, 1 death
        # (living target count): target birth, 1 death


        priors = []
        likelihoods = []
        events = []

        rem_meas = len(measurement_list) - index #number of remaining measurements
        #No target deaths
        no_death_prior = np.prod(np.ones(len(p_target_deaths))-np.array(p_target_deaths))
        #
        # Priors with no target deaths
        #
        pt = 0 # One of targets
        pn = 0 # New target
        pc = 0 # Clutter
        for t in range(0,min(unas_living_targ_count,rem_meas)+1): #t of the remaining measurements are associated with a target
            for n in range(0,rem_meas-t+1): #n of the remaining measurements are associated with a birth
                #probability of t target associations and n measurement
                #associations
                pp = bin_pdf(n,pb,rem_meas-t) * bin_pdf(t,prob_detection,min(unas_living_targ_count,rem_meas))
                pt = pt + float(t)/rem_meas * pp
                pn = pn + float(n)/rem_meas * pp
                pc = pc + float(rem_meas-t-n)/rem_meas * pp

        #clutter association
        priors.append(no_death_prior*pc)
        likelihoods.append(CD)
        events.append(SampledEvent(assoc_index=-1, dead_targ_idx=None))

        #living target association
        check_living_unassoc_targ_count = 0
        for target_index in range(total_target_count):
            if((not target_index in list_of_measurement_associations) and \
               (not target_index in indices_to_kill)):
                check_living_unassoc_targ_count += 1
                cur_target_likelihood = memoized_assoc_likelihood(particle, cur_meas, meas_source_index,\
                                                                  target_index, params, meas_score)
                priors.append(no_death_prior*pt/unas_living_targ_count)
                likelihoods.append(cur_target_likelihood)
                events.append(SampledEvent(assoc_index=target_index, dead_targ_idx=None))
        assert(check_living_unassoc_targ_count == unas_living_targ_count), (check_living_unassoc_targ_count, unas_living_targ_count)
        #new target
        priors.append(no_death_prior*pn)
        likelihoods.append(birth_likelihood)
        events.append(SampledEvent(assoc_index=total_target_count, dead_targ_idx=None))
        #
        # Priors with 1 target death
        #
        pt = 0 # One of targets
        pn = 0 # New target
        pc = 0 # Clutter
        for t in range(0,min(unas_living_targ_count-1,rem_meas)+1): #t of the remaining measurements are associated with a target
            for n in range(0,rem_meas-t+1): #n of the remaining measurements are associated with a birth
                #probability of t target associations and n measurement
                #associations
                pp = bin_pdf(n,pb,rem_meas-t) * bin_pdf(t,prob_detection,min(unas_living_targ_count-1,rem_meas))
                pt = pt + float(t)/rem_meas * pp
                pn = pn + float(n)/rem_meas * pp
                pc = pc + float(rem_meas-t-n)/rem_meas * pp

        #
        # Association to clutter, target with index target_index dies
        #
        check_living_targ_count = 0
        for target_index in range(total_target_count):
            if((not target_index in indices_to_kill)):
                check_living_targ_count += 1
                cur_death_prior = no_death_prior/(1.0 - p_target_deaths[target_index])*p_target_deaths[target_index]
                priors.append(cur_death_prior*pc)
                likelihoods.append(CD)
                events.append(SampledEvent(assoc_index=-1, dead_targ_idx=target_index))
        assert(check_living_targ_count == cur_living_targ_count), (check_living_targ_count, cur_living_targ_count)

        #
        # Association to target assoc_target_ind, target dead_target_idx dies
        #
        check_living_unassoc_targ_count = 0
        for assoc_target_ind in range(total_target_count):
            if((not assoc_target_ind in list_of_measurement_associations) and \
               (not assoc_target_ind in indices_to_kill)):
                check_living_unassoc_targ_count += 1
                for dead_target_idx in range(total_target_count):
                    if (not dead_target_idx in indices_to_kill) and (dead_target_idx != assoc_target_ind):
                        cur_death_prior = no_death_prior/(1.0 - p_target_deaths[dead_target_idx])*p_target_deaths[dead_target_idx]
                        priors.append(cur_death_prior*pt/unas_living_targ_count)
                        cur_target_likelihood = memoized_assoc_likelihood(particle, cur_meas, meas_source_index,\
                                                                          assoc_target_ind, params, meas_score)                        
                        likelihoods.append(cur_target_likelihood)
                        events.append(SampledEvent(assoc_index=assoc_target_ind, dead_targ_idx=dead_target_idx))
        assert(check_living_unassoc_targ_count == unas_living_targ_count), (check_living_unassoc_targ_count, unas_living_targ_count)

        #
        # Association to new target, target with index target_index dies
        #
        check_living_targ_count = 0
        for target_index in range(total_target_count):
            if((not target_index in indices_to_kill)):
                check_living_targ_count += 1
                cur_death_prior = no_death_prior/(1.0 - p_target_deaths[target_index])*p_target_deaths[target_index]
                priors.append(cur_death_prior*pn)
                likelihoods.append(birth_likelihood)
                events.append(SampledEvent(assoc_index=total_target_count, dead_targ_idx=target_index))
        assert(check_living_targ_count == cur_living_targ_count), (check_living_targ_count, cur_living_targ_count)


        #create proposal distribution for the current measurement with the following entries:
        # 1: clutter, no deaths
        # (unassociated, living target count): target association, no deaths
        # 1: target birth, no deaths
        #
        # (living target count): clutter association, 1 death
        # (unassociated, living target count)*(living target count - 1): target association, 1 death
        # (living target count): target birth, 1 death

        #Sample
        assert(len(priors) == len(likelihoods))
        assert(len(priors) == len(events))
        assert(len(priors) == (2 + unas_living_targ_count + 2*cur_living_targ_count + unas_living_targ_count*(cur_living_targ_count-1))), (len(priors), unas_living_targ_count, cur_living_targ_count, total_target_count)
        priors = np.asarray(priors)
        assert(float(np.sum(priors)) > 0), (priors, total_target_count, remaining_meas_count, len(measurement_list), cur_living_targ_count, unas_living_targ_count)
        priors /= float(np.sum(priors))
        original_proposal_distr = np.multiply(priors, likelihoods)   
        assert(float(np.sum(original_proposal_distr)) > 0), (priors, total_target_count, remaining_meas_count, len(measurement_list), cur_living_targ_count, unas_living_targ_count)             
        original_proposal_distr /= float(np.sum(original_proposal_distr))


        sampled_assoc_idx = np.random.choice(len(original_proposal_distr),
                                                p=original_proposal_distr)


        importance_reweight = importance_reweight*likelihoods[sampled_assoc_idx]*\
                              priors[sampled_assoc_idx]/original_proposal_distr[sampled_assoc_idx]

        #process the sampled event
        assoc_index = events[sampled_assoc_idx].assoc_index
        if(assoc_index in list_of_measurement_associations):
            assert(assoc_index == -1 or assoc_index == total_target_count)
        if(assoc_index>=0 and assoc_index < total_target_count):
            unas_living_targ_count -= 1
            p_target_deaths[assoc_index] = death_prob_of_assoc_target(default_time_step)
        list_of_measurement_associations.append(assoc_index)

        death_index = events[sampled_assoc_idx].dead_targ_idx
        assert(not death_index in indices_to_kill), (death_index, indices_to_kill)
        if(death_index):
            cur_living_targ_count -= 1
            if(not death_index in list_of_measurement_associations):
                unas_living_targ_count -= 1
            p_target_deaths[death_index] = 0.0
            indices_to_kill.append(death_index)

#        #debugging
#        if(particle.max_importance_weight):
#            if(assoc_index == -1):
#                print "measurement", index, "associated with clutter"
#            elif(assoc_index == total_target_count):
#                print "measurement", index, "associated with a new target"
#            else:
#                print "measurement", index, "associated with target", assoc_index
#
#            if(death_index):
#                print "target", death_index, "died"
#            else:
#                print "no targets died"
#
#        #end debugging

        remaining_meas_count -= 1

#        assert(clutter_count <= params.max_clutter_count(meas_source_index, meas_score))
#        assert(birth_count <= params.max_birth_count(meas_source_index, meas_score)), (birth_count, params.max_birth_count(meas_source_index, meas_score), index, remaining_meas_count, len(proposal_distribution), proposal_distribution, birth_count, clutter_count, len(measurement_list), total_target_count)

    assert(remaining_meas_count == 0)
    return(list_of_measurement_associations, indices_to_kill, importance_reweight)


def sample_target_deaths(particle, unassociated_targets, cur_time):
    """
    Sample target deaths, given they have not been associated with a measurement, using probabilities
    learned from data.
    Also kill all targets that are offscreen.

    Inputs:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - unassociated_targets: a list of target indices that have not been associated with a measurement

    Output:
    - targets_to_kill: a list of targets that have been sampled to die (not killed yet)
    - probability_of_deaths: the probability of the sampled deaths
    """
    targets_to_kill = []
    probability_of_deaths = 1.0

    for target_idx in range(len(particle.targets.living_targets)):
        #kill offscreen targets with probability 1.0
        if(particle.targets.living_targets[target_idx].offscreen == True):
            targets_to_kill.append(target_idx)
        elif(target_idx in unassociated_targets):
            cur_death_prob = particle.targets.living_targets[target_idx].death_prob
            if(random.random() < cur_death_prob):
                targets_to_kill.append(target_idx)
                probability_of_deaths *= cur_death_prob
            else:
                probability_of_deaths *= (1 - cur_death_prob)
    return (targets_to_kill, probability_of_deaths)

def calc_death_prior(living_target_indices, p_target_deaths):
    death_prior = 1.0
    for (cur_target_index, cur_target_death_prob) in enumerate(p_target_deaths):
        if cur_target_index in living_target_indices:
            death_prior *= (1.0 - cur_target_death_prob)
            assert((1.0 - cur_target_death_prob) != 0.0), cur_target_death_prob
        else:
            death_prior *= cur_target_death_prob
            assert((cur_target_death_prob) != 0.0), cur_target_death_prob

    return death_prior


def memoized_assoc_likelihood(particle, measurement, meas_source_index, target_index, params, meas_score):
    """
        LSVM and regionlets produced two measurements with the same locations (centers), so using the 
        meas_source_index as part of the key is (sort of) necessary.  Currently also using the score_index, 
        could possibly be removed (not sure if this would improve speed).

        Currently saving more in the value than necessary (from debugging), can eliminate to improve
        performance (possibly noticable)

    Inputs:
    - params: type Parameters, gives prior probabilities and other parameters we are using

    """


    if((measurement[0], measurement[1], target_index, meas_source_index, meas_score) in particle.assoc_likelihood_cache):
        (assoc_likelihood, cached_score_index, cached_measurement, cached_meas_source_index) = particle.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, meas_source_index, meas_score)]
        assert(cached_score_index == meas_score), (cached_score_index, meas_score, measurement, cached_measurement, target_index, meas_noise_cov, cached_meas_source_index, meas_source_index)
        assert(cached_meas_source_index == meas_source_index), (cached_score_index, meas_score, measurement, cached_measurement, target_index, meas_noise_cov, cached_meas_source_index, meas_source_index)
        return assoc_likelihood
    else: #likelihood not cached
        R = params.get_R(meas_source_index, meas_score)
        target = particle.targets.living_targets[target_index]
        S = np.dot(np.dot(params.H, target.P), params.H.T) + R
        assert(target.x.shape == (4, 1))

        state_mean_meas_space = np.dot(params.H, target.x)
        state_mean_meas_space = np.squeeze(state_mean_meas_space)


        if params.USE_PYTHON_GAUSSIAN:
            distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)
            assoc_likelihood = distribution.pdf(measurement)
        else:
            S_det = S[0][0]*S[1][1] - S[0][1]*S[1][0] # a little faster
            S_inv = inv(S)
            LIKELIHOOD_DISTR_NORM = 1.0/math.sqrt((2*math.pi)**2*S_det)

            offset = measurement - state_mean_meas_space
            a = -.5*np.dot(np.dot(offset, S_inv), offset)
            assoc_likelihood = LIKELIHOOD_DISTR_NORM*math.exp(a)

        particle.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, meas_source_index, meas_score)] = (assoc_likelihood, meas_score, measurement, meas_source_index)
        return assoc_likelihood


