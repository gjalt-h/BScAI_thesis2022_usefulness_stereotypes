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
def res_wag_update(assoc, comp_val, present, alpha = 0.07, beta1 = 0.4, beta2 = 0.1, v_max = 1):
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
def init_aliens(n_att, total_n_att, n_feat): 
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
        attributes_ass.append(np.random.choice(range(total_n_att), size=n_att))
    
    return(aliens, attributes_ass)

aliens = init_aliens(6,48,3)[0]


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


def experiment(n_sequences=100, sequence_len=7, views=3, recall=.8,
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
                            
                            alpha_att = 0.07
                            V_att = 0.5
                            # with colour salience ---------------------------
                            if feat == 0:
                                alpha_att = alpha_att * 1.3
                                V_att = V_att * 2
                            # ------------------------------------------------

                            curr_assoc = associations[feat][alien_var][att] 
                            new_assoc = res_wag_update(curr_assoc, comp_value, present = presence, alpha = alpha_att, v_max = V_att)
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


# unique evalutation -------------------------------------------------
# function to count unique occurences of attributes
def unique_occ(sequence):
    unique = []

    # create list of all attributes used per iteration
    for participant in sequence:
        total_list = []
        for alien in participant:
            total_list += list(alien)

        # convert to set to remove duplicates and count
        counter = len(set(total_list))
        unique.append(counter)

    return unique
        
# test the first sequence
sequence1 = results[0]

# create list with unique occurences to use in z_score calculations
def unique_list(results):
    # create list with all uniqueness scores per sequence
    uniqueness_results = []
    for sequence in results:
        uniqueness_results.append(unique_occ(sequence))

    # create list with mean uniqueness scores per sequence
    uniqueness_means = []
    for i in range(len(results[0])):
        sum = 0
        for j in range(len(results)):
            sum += uniqueness_results[j][i]
        uniqueness_means.append(sum / len(results))

    return uniqueness_results, uniqueness_means

unique_res, unique_mean = unique_list(results)


def eval_unique_err(sequences):
    means, errors = sequence_to_mean_error(sequences)
    plt.errorbar(range(len(means)), means, yerr=errors, capsize=5, elinewidth=.5, capthick=.5)
    plt.xlabel('iteration')
    plt.ylim((6,48))
    plt.title('Simplification with error bars')
    plt.savefig('graphs/unique_error.jpg')
    plt.close()

eval_unique_err(unique_res)


# overlap -------------------------------------------------
def eval_overlap(sequence, aliens) -> list:
    n_feat = len(aliens[0])

    overlap_scores = []
    overlap_means = []

    for iteration in sequence:
        overlap_per_feats_com = [[] for _ in range(n_feat)]
        
        # compare every alien to other aliens
        for alien_id in range(len(iteration)):
            for other_id in range(len(iteration)):
                if alien_id is not other_id:

                    # calculate overlap
                    overlap = len(set(iteration[alien_id]).intersection(iteration[other_id]))
                    
                    # determine features in common
                    feats_com = 0
                    for i in range(n_feat):                  
                        if aliens[alien_id][i] == aliens[other_id][i]:
                            feats_com += 1
                    
                    # append to correct
                    overlap_per_feats_com[feats_com].append(overlap)
        
        overlap_scores.append(overlap_per_feats_com)

        # create mean overlap list
        overlap_means_it = []
        for i in range(n_feat):
            overlap_means_it.append(np.mean(overlap_per_feats_com[i]))
        overlap_means.append(overlap_means_it)

    # convert to sequentually per feature
    overlap_means_per_feat = [[] for _ in range(n_feat)]
    for feat in range(n_feat):
        for means in overlap_means:
            overlap_means_per_feat[feat].append(means[feat])
    
    return overlap_scores, overlap_means_per_feat, overlap_means

# for 1 sequence!

def overlap_all(results, aliens):
    overlap_scores = []
    overlap_means_feat = []
    overlap_means_it = []

    for sequence in results:
        sc, mf, mi = eval_overlap(sequence, aliens)
        overlap_scores.append(sc)
        overlap_means_feat.append(mf)
        overlap_means_it.append(mi)
    
    return overlap_scores, overlap_means_feat, overlap_means_it

overlap_scores_all, overlap_means_feat_all, overlap_means_it_all = overlap_all(results, aliens)


def feat_all_to_mean(all_overlap_means_feat):
    n_seq = len(all_overlap_means_feat)
    n_feat = len(all_overlap_means_feat[0])
    n_it = len(all_overlap_means_feat[0][0])
    means = [[] for _ in range(n_feat)]
    for feat in range(n_feat):
        chain = []
        for i in range(n_it):
            sum = 0
            for seq in range(n_seq):
                sum += all_overlap_means_feat[seq][feat][i]
            chain.append(sum / n_seq)
        means[feat] = chain
    return means

overlap_means_feat = feat_all_to_mean(overlap_means_feat_all)


# read monte carlo files into a Python list object ---------
with open('monte_carlos_means.txt', 'r') as f:
    mcs_m = json.loads(f.read())

with open('monte_carlos_stds.txt', 'r') as g:
    mcs_std = json.loads(g.read())
# ----------------------------------------------------------


# convert overlap scores to z_scores (uniqueness value necessary and monte carlos values)
def convert_overlap_z(overlap_means, mcs_m, mcs_std, uniqueness):
    zs = [[] for _ in range(len(overlap_means))]
    for feat in range(len(overlap_means)):
        z_feat = []
        for i in range(len(overlap_means[0])):
            unique_i = int(uniqueness[i]) #int because avg
            mc_mean = mcs_m[unique_i]
            mc_std = mcs_std[unique_i]

            # calculate z score
            z = (overlap_means[feat][i] - mc_mean) / mc_std

            z_feat.append(z)

        zs[feat] = z_feat
    return zs

# generate z_scores
z_scores_avg = convert_overlap_z(overlap_means_feat, mcs_m, mcs_std, unique_mean)


# generate z_scores per sequence for error plot
def z_scores_sequence(results, aliens, mcs_m, mcs_std):
    overlaps_all = overlap_all(results, aliens)[1]
    uniqueness = unique_list(results)[0]
    z_scores_sequences = []
    for i in range(len(uniqueness)):
        z_scores_sequences.append(convert_overlap_z(overlaps_all[i], mcs_m, mcs_std, uniqueness[i]))
    return z_scores_sequences

z_scores_sequences = z_scores_sequence(results, aliens, mcs_m=mcs_m, mcs_std=mcs_std)

def z_scores_sequence_to_feat(z_scores):
    z_scores_feat = []
    for feat in range(len(z_scores[0])):
        z_scor = []
        for zs in z_scores:
            z_scor.append(zs[feat])
        z_scores_feat.append(z_scor)
    return z_scores_feat

z_scores_feat = z_scores_sequence_to_feat(z_scores_sequences)


def eval_overlap_z_err(feat_sequences):
    for feat in feat_sequences:
        means, errors = sequence_to_mean_error(feat)
        plt.errorbar(range(len(means)), means, yerr=errors, capsize=5, elinewidth=.5, capthick=.5)
    plt.xlabel('iteration')
    plt.ylabel('structure z-score')
    plt.axhline(y=1.96, color='r', linestyle='-')
    plt.title('overlap z-score per feat with error')
    plt.legend(['1.96 chance', '0 features in common', '1 feature in commmon', '2 features in common'])
    plt.savefig('graphs/overlap_z_error.jpg')
    plt.close()

eval_overlap_z_err(z_scores_feat)


# overlap per feat_type (color, movement, shape) -------------------------------------------------
def eval_overlap_col(sequence, aliens) -> list:
    n_feat = len(aliens[0])

    overlap_scores = []
    overlap_means = []

    for iteration in sequence:
        overlap_per_feats = [[] for _ in range(n_feat)]
        
        # compare every alien to other aliens
        for alien_id in range(len(iteration)):
            for other_id in range(len(iteration)):
                if alien_id is not other_id:

                    # calculate overlap
                    overlap = len(set(iteration[alien_id]).intersection(iteration[other_id]))
                    
                    # determine if features in common !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    for i in range(n_feat):                  
                        if aliens[alien_id][i] == aliens[other_id][i]:
                            # append to correct
                             overlap_per_feats[i].append(overlap)
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        overlap_scores.append(overlap_per_feats)

        # create mean overlap list
        overlap_means_it = []
        for i in range(n_feat):
            overlap_means_it.append(np.mean(overlap_per_feats[i]))
        overlap_means.append(overlap_means_it)

    # convert to sequentually per feature
    overlap_means_per_feat = [[] for _ in range(n_feat)]
    for feat in range(n_feat):
        for means in overlap_means:
            overlap_means_per_feat[feat].append(means[feat])
    
    return overlap_scores, overlap_means_per_feat, overlap_means

def overlap_all_col(results, aliens):
    overlap_scores = []
    overlap_means_feat = []
    overlap_means_it = []

    for sequence in results:
        sc, mf, mi = eval_overlap_col(sequence, aliens)
        overlap_scores.append(sc)
        overlap_means_feat.append(mf)
        overlap_means_it.append(mi)
    
    return overlap_scores, overlap_means_feat, overlap_means_it

col_overlap_scores_all, col_overlap_means_feat_all, col_overlap_means_it_all = overlap_all_col(results, aliens)

col_overlap_means_feat = feat_all_to_mean(col_overlap_means_feat_all)

# with z_score
# generate z_scores per sequence for error plot
def z_scores_sequence_col(results, aliens, mcs_m, mcs_std):
    overlaps_all = overlap_all_col(results, aliens)[1]
    uniqueness = unique_list(results)[0]
    z_scores_sequences = []
    for i in range(len(uniqueness)):
        z_scores_sequences.append(convert_overlap_z(overlaps_all[i], mcs_m, mcs_std, uniqueness[i]))
    return z_scores_sequences

col_z_scores_sequences = z_scores_sequence_col(results, aliens, mcs_m=mcs_m, mcs_std=mcs_std)

col_z_scores_feat = z_scores_sequence_to_feat(col_z_scores_sequences)

def eval_overlap_z_err_col(feat_sequences):
    for feat in feat_sequences:
        means, errors = sequence_to_mean_error(feat)
        plt.errorbar(range(len(means)), means, yerr=errors, capsize=5, elinewidth=.5, capthick=.5)
    plt.xlabel('iteration')
    plt.ylabel('structure z-score')
    plt.axhline(y=1.96, color='r', linestyle='-')
    plt.title('overlap z-score per feat with error')
    plt.legend(['1.96 chance', 'colour', 'shape', 'movement'])
    plt.savefig('graphs/col_overlap_z_error.jpg')
    plt.close()

eval_overlap_z_err_col(col_z_scores_feat)
