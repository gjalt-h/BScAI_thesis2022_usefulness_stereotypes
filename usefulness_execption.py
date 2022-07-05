# ---------------------------------------
# Afstudeerproject BSc KI
# stereotype formation: cognitive modeling
# Date: May 2nd 2022
# Supervisor: Katrin Schulz
# 
# Description: Base model simulating the
# formation of stereotypes as researched
# by Martin et al. (2014). 
# ---------------------------------------

import numpy as np
import random
import matplotlib.pyplot as plt
import json
import scipy.stats as st

# own implementation of rescorla wagner update
def res_wag_update(assoc, comp_val, present, alpha = 0.1, beta1 = 0.4, beta2 = 0.1, v_max = 1):
    # compute the prediction error
    if present is True: 
        error_pred = v_max - comp_val 
        beta = beta1  

    elif present is False:
        error_pred = 0 - comp_val
        beta = beta2

    # compute the change in strength
    change = alpha * beta * error_pred

    # update the association value
    assoc = assoc + change

    # return the new value
    return assoc


def limited_recall(prob_dist, threshold):
    # distribution save indexes and values
    indexes = []
    values = []
    sum = 0

    # repeat for each index
    for _ in prob_dist:

        # find max and save index and value
        max_prob_i = np.argmax(prob_dist)
        indexes.append(max_prob_i)
        values.append(prob_dist[max_prob_i])

        # update sum and set zero for next iteration
        sum += prob_dist[max_prob_i]
        prob_dist[max_prob_i] = 0

        # check threshold
        if sum >= threshold:
            break

    # create limited recall distribution
    limited = [0] * len(prob_dist)

    # loop over indexes and values and add
    for i in range(len(indexes)):
        limited[indexes[i]] = values[i]
    
    # normalise distribution
    normalizer = np.sum(limited)
    for j in range(len(limited)):
        limited[j] = limited[j] / normalizer
    
    return limited


# 0 is color, 1 is shape, 2 is movement 
def init_aliens(n_att, total_n_att, n_feat, ratio=1): # ratio is between groups, so actually double for useful:others
    # create unique aliens with n features
    n_aliens = n_feat**(n_feat)
    aliens = {}
    for i in range(n_aliens):
        color = i // n_feat**2
        shape = (i // n_feat) % 3
        move = i % 3 
        aliens[i] = [color, shape, move]
    
    # for each alien create assigned attribute list
    attributes_ass = []

    for _ in range(n_aliens):
        attributes_ass.append(np.random.choice(range(total_n_att-1), size=n_att, replace=False))

    # create 1 exception
    attributes_ass[26][5] = 47
    
    return(aliens, attributes_ass)

aliens = init_aliens(6,48,3)[0]

# higher attribute is higher usefulness # alpha 0.07, V=0.8
def division_useful(att, alpha=0.07, V=0.5, alpha_extra=0.1, V_extra=1.2):
    if att == 47:
        return alpha_extra, V_extra
    else:
        return alpha, V


# create initial associations dictionary with base values
def init_associations(n_feat, n_var, total_n_att):
    assoc_dict = {}

    # entry for every variant of each feature
    for feature in range(n_feat):
        assoc_dict[feature] = {}
        for var in range(n_var):

            # initialize with base association for every attribute
            assoc_dict[feature][var] = [1/total_n_att] * total_n_att

            assoc_dict[feature][var] = []
            for _ in range(total_n_att):
                assoc_dict[feature][var].append(np.random.uniform(0,(1/total_n_att)))

    return assoc_dict


def experiment(n_sequences=20, sequence_len=7, views=3, recall=.8,
            train_size=13, n_att=6, total_n_att=48, n_feat=3, n_var=3):
    # setting up the experiment -----------------------    
    results = []

    # repeat for every sequence
    for _ in range(n_sequences):
        # list for assignments per sequence
        sequence_ass = []

        # generate aliens and assigned attributes
        aliens, att_ass = init_aliens(n_att, total_n_att, n_feat)
        sequence_ass.append(att_ass)

    # actual experiment ----------------------------
        # iterate n times per sequence
        for _ in range(sequence_len):

            # create starting dictionary of associations
            associations = init_associations(n_feat, n_var, total_n_att)

            # training phase 

            # create training set
            train_ids = random.sample(range(len(aliens)), train_size)
            train_set = {}
            for id in train_ids:
                train_set[id] = aliens[id]

            # run rescorla wagner update n times

            for _ in range(views):
                for alien_id, alien_feat in train_set.items():

                    # own implementation-------------------------------
                    # run for every attribute
                    for att in range(total_n_att):

                        # determine presence
                        if att in att_ass[alien_id]:
                            presence = True
                        elif att not in att_ass[alien_id]:
                            presence = False

                        # compute computational value (sum of CS associations)
                        comp_value = 0
                        for feat, alien_var in enumerate(alien_feat):
                            comp_value += associations[feat][alien_var][att]

                        # run R-W formula
                        for feat, alien_var in enumerate(alien_feat):
                            
                            alpha_att, V_att = division_useful(att)

                            # with colour salience ---------------------------
                            if feat == 0:
                                alpha_att = alpha_att * 1.3
                                V_att = V_att * 2
                            # ------------------------------------------------

                            curr_assoc = associations[feat][alien_var][att] 
                            new_assoc = res_wag_update(curr_assoc, comp_value, present = presence, alpha=alpha_att, v_max=V_att)
                            associations[feat][alien_var][att] = new_assoc
                    # -----------------------------------------------------

            # test phase
            # list for assignment per participant
            alien_output = []

            # assign new attributes for aliens
            for key in aliens: # aliens is a dict
                alien_assocs= []

                # find the corresponding associations
                for feat in range(n_feat):
                    alien_assocs.append(associations[feat][aliens[key][feat]])
                
                # calculate mean for each attribute
                combined_assoc = []
                for att in range(total_n_att):
                    sum = 0
                    for feat in range(n_feat):
                        sum += alien_assocs[feat][att]
                    combined_assoc.append(sum / n_feat)

                prob_dist = [(assoc / np.sum(combined_assoc)) for assoc in combined_assoc]

                # apply limited recall here
                prob_dist = limited_recall(prob_dist,recall)
                # --------------------------------------------------------------------

                # pick from retrieved prob dist
                alien_ass = np.random.choice(range(total_n_att), size=n_att, p=prob_dist, replace=False)
                
                # append to alien output for this participant 
                alien_output.append(alien_ass)
            
            # append assignments to sequence list + output becomes input next
            sequence_ass.append(alien_output)
            att_ass = alien_output

        # append to total results list
        results.append(sequence_ass)
        print('sequence')
    
    # returns list of the alien assignment of each iteration
    return results

# run experiment
results = experiment()


# mean error functionality ------------------------------------
def sequence_to_mean_error(sequences, ci=0.95):
    means = []
    lowers = []
    uppers = []
    for i in range(len(sequences[0])):
        data = []
        for sequence in sequences:
            data.append(sequence[i])
        means.append(np.mean(data))
        data = np.array(data)
        lower, upper = st.t.interval(alpha=ci, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
        lowers.append(lower)
        uppers.append(upper)
    errors = [abs(np.array(means) - np.array(lowers)), abs(np.array(means) - np.array(uppers))]
    return means, errors


# evaluate 1 exception 47.
def count_exception_seq(sequence):
    counters = []
    for iteration in sequence:
        counter = 0
        for alien in iteration:
            if 47 in alien:
                counter += 1
        counters.append(counter)
    return counters

def count_exception_all(results):
    all_counters = []
    for sequence in results:
        all_counters.append(count_exception_seq(sequence))
    return all_counters

def eval_exception(results):
    all_counters = count_exception_all(results)
    for counters in all_counters:
        plt.plot(range(8), counters)
    plt.title('exception graph')
    plt.xlabel('iteration')
    plt.ylabel('occurence of exception')

    plt.savefig('graphs/exception_graph.jpg')
    plt.close()

eval_exception(results)
