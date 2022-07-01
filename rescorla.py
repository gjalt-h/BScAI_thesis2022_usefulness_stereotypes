# ---------------------------------------
# Afstudeerproject BSc KI
# stereotype formation: cognitive modeling
# Date: May 2nd 2022
# Supervisor: Katrin Schulz
# 
# Description: creating a Rescorla wagner 
# model to assess usefulness influence on 
# formation of stereotypes as researched
# by Martin et al. (2014).
# ---------------------------------------

import numpy as np
import random
import matplotlib.pyplot as plt
import json
# from prometheus_client import Counter
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
    
    # normalize distribution
    normalizer = np.sum(limited)
    for j in range(len(limited)):
        limited[j] = limited[j] / normalizer
    
    return limited


# 0 is color, 1 is shape, 2 is movement  # extra ratio is for 1 exception, the last attribute 48.
def init_aliens(n_att, total_n_att, n_feat, ratio=1, extra_ratio=20): # ratio is between groups, so actually double for useful:others
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

    # create prob dist for sampling
    prob_dist = []
    for group in range(3):
        if group < 2:
            for _ in range(16):
                prob_dist.append(ratio)
        elif group == 2:
            for _ in range(16):
                prob_dist.append(1)
            prob_dist[-1] = 0 #  0 means that exception only occurs with 1 group
            # prob_dist.append(1/extra_ratio) 
    total = sum(prob_dist)
    for j in range(len(prob_dist)):
        prob_dist[j] = prob_dist[j] / total

    for _ in range(n_aliens):
        attributes_ass.append(np.random.choice(range(total_n_att), size=n_att, replace=False, p=prob_dist))

    # create 1 exception
    attributes_ass[26][5] = 47
    
    return(aliens, attributes_ass)

aliens = init_aliens(6,48,3)[0]

# higher attribute is higher usefulness # alpha 0.07, V=0.8
def division_useful(att, total_n_att=48, alpha_less=0.05, alpha=0.1,
                alpha_more=0.1, V_less=0.25, V=1, V_more=1, alpha_extra=0.1, V_extra=1.2):
    # if att < total_n_att // 3:
    #     return alpha_less, V_less
    # elif att < 2*total_n_att // 3:
    #     return alpha, V
    # else:
    #     # if att == 47:
    #     #     return alpha_extra, V_extra
    #     return alpha_more, V_more

    # # for now without division for easy change
    # if att == 47:
    #     return alpha_extra, V_extra

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

        # # create starting dictionary of associations
        # associations = init_associations(n_feat, n_var, total_n_att)

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
                
                # # new idea: first recall, then average -----------------------
                # prob_dists = []
                # for assocs in alien_assocs:
                #     prob_dists.append([(assoc / np.sum(assocs)) for assoc in assocs])

                # lim_prob_dists = []
                # for probs in prob_dists:
                #     lim_prob_dists.append(limited_recall(probs,recall))

                # prob_dist = []
                # for att in range(total_n_att):
                #     sum = 0
                #     for feat in range(n_feat):
                #         sum += lim_prob_dists[feat][att]
                #     prob_dist.append(sum / n_feat)
                # # ---------------------------------------------------------------

                # first implementation (combine and then lim recall) ---------------
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

                # eerst prob dist dan combi dan recall uitproberen, kijken wat er dan gebeurt
                # 

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
        # from https://www.statology.org/confidence-intervals-python/
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


def eval_unique(results):
    # plot each sequence
    for sequence in results:
        plt.plot(range(len(sequence)), unique_occ(sequence))

    # create plot
    plt.xlabel('iteration')
    plt.ylabel('n unique attributes')
    plt.ylim((6,48))
    plt.title('Simplification over iterations')
    plt.savefig('../graphs/uniqueness.jpg')
    plt.close()

# # run unique graphs result
eval_unique(results)

def eval_unique_err(sequences):
    means, errors = sequence_to_mean_error(sequences)
    plt.errorbar(range(len(means)), means, yerr=errors, capsize=5, elinewidth=.5, capthick=.5)
    plt.xlabel('iteration')
    plt.ylim((6,48))
    plt.title('Simplification with error bars')
    plt.savefig('../graphs/unique_error.jpg')
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

def plt_overlap(overlap_means):
    for feat in overlap_means:
        plt.plot(range(len(overlap_means[0])),feat)
    plt.title('overlap graph')
    plt.xlabel('iteration')
    plt.ylabel('average overlap')
    plt.legend(['0 features in common', '1 feature in commmon', '2 features in common'])
    plt.savefig('../graphs/overlap_graph1.jpg')
    plt.close()

plt_overlap(overlap_means_feat)


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


# plot the z_scores
def plt_overlap_z(overlap_zs):
    for feat in overlap_zs:
        plt.plot(range(len(overlap_zs[0])),feat)
    plt.axhline(y=1.96, color='r', linestyle='-')
    plt.title('overlap graph z-score')
    plt.xlabel('iteration')
    plt.ylabel('average overlap z-score')
    plt.legend(['0 features in common', '1 feature in commmon', '2 features in common'])
    plt.savefig('../graphs/overlap_graph_z_av.jpg')
    plt.close()

plt_overlap_z(z_scores_avg)

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
    plt.savefig('../graphs/overlap_z_error.jpg')
    plt.close()

eval_overlap_z_err(z_scores_feat)



# usefulness with attributes ----------------------------------

# higher att is more useful, so 0 is less, 1 is normal, 2 is more
def att_occ_counter(iteration):
    counters= [0,0,0]
    for alien in iteration:
        for attribute in alien:
            if attribute < 16:
                counters[0] += 1
            elif attribute < 32:
                counters[1] += 1
            else:
                counters[2] += 1
    return counters

def att_occ_sequence(sequence):
    counters_grouped = [[] for _ in range(3)]
    for iteration in sequence:
        it_counters = att_occ_counter(iteration)
        for i in range(3):
            counters_grouped[i].append(it_counters[i])
    return counters_grouped

def eval_att_occ_useful(results):
    att_occ_all_counters = [[] for _ in range(3)]
    for sequence in results:
        for i in range(3):
            att_occ_all_counters[i].append(att_occ_sequence(sequence)[i])
    for counters_group in att_occ_all_counters:
        means, errors = sequence_to_mean_error(counters_group)
        plt.errorbar(range(len(means)), means, yerr=errors, capsize=5, elinewidth=.5, capthick=.5)
    plt.ylabel('total attributes')
    plt.xlabel('iteration')
    plt.title('total number of attributes per usefulness group')
    plt.legend(['low usefulness', 'normal usefulness', 'high usefulness'])
    plt.savefig('../graphs/usefulness_total_occ.jpg')
    plt.close()

eval_att_occ_useful(results)

def unique_att_grouped_useful(iteration):
    att_groups = [[] for _ in range(3)]
    for alien in iteration:
        for attribute in alien:
            if attribute < 16:
                att_groups[0].append(attribute)
            elif attribute < 32:
                att_groups[1].append(attribute)               
            else:
                att_groups[2].append(attribute)
    return att_groups

def unique_att_grouped_sequence(sequence):
    att_grouped = [[] for _ in range(3)]
    for iteration in sequence:
        it_grouped = unique_att_grouped_useful(iteration)
        for i in range(3):
            att_grouped[i].append(it_grouped[i])
    return att_grouped

def unique_att_grouped_counted(sequence):
    att_grouped = unique_att_grouped_sequence(sequence)
    groups_counted = [[] for _ in range(3)]
    for i in range(3):
        counters = []
        for iteration in att_grouped[i]:
            counters.append(len(set(iteration)))
        groups_counted[i] = counters
    return groups_counted

def eval_unique_att_count(results):
    unique_att_all_counters = [[] for _ in range(3)]
    for sequence in results:
        for i in range(3):
            unique_att_all_counters[i].append(unique_att_grouped_counted(sequence)[i])
    for counters_group in unique_att_all_counters:
        means, errors = sequence_to_mean_error(counters_group)
        plt.errorbar(range(len(means)), means, yerr=errors)

    plt.ylabel('unique attributes')
    plt.xlabel('iteration')
    plt.title('unique number of attributes per usefulness group')
    plt.legend(['low usefulness', 'normal usefulness', 'high usefulness'])
    plt.savefig('../graphs/usefulness_unique_att.jpg')

    plt.close()

eval_unique_att_count(results)



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

    plt.savefig('../graphs/exception_graph1.jpg')
    plt.close()

eval_exception(results)



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

def plt_overlap_col(overlap_means):
    for feat in overlap_means:
        plt.plot(range(len(overlap_means[0])),feat)
    plt.title('overlap graph')
    plt.xlabel('iteration')
    plt.ylabel('average overlap')
    plt.legend(['color', 'shape', 'movement'])
    plt.savefig('../graphs/color_overlap_graph1.jpg')
    plt.close()

plt_overlap_col(col_overlap_means_feat)

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
    plt.savefig('../graphs/col_overlap_z_error.jpg')
    plt.close()

eval_overlap_z_err_col(col_z_scores_feat)
