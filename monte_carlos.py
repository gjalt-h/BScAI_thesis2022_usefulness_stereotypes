# ---------------------------------------
# Afstudeerproject BSc KI
# stereotype formation: cognitive modeling
# Date: July 1st 2022
# Supervisor: Katrin Schulz
# 
# Description: monte carlo trials for 
# normalising structure scores.
# ---------------------------------------

import numpy as np
import random
import json

def monte_carlo(total_n_att, n_aliens = 27, n_att=6, trials=12000):
    overlap_means = []
    for _ in range(trials):
        assigns = []
        # assign random attributes to aliens
        for _ in range(n_aliens):
            assign = np.random.choice(range(total_n_att), replace=False, size=n_att)
            assigns.append(assign)
        # calculate avg overlap
        overlaps = []
        for i in range(n_aliens):
            for j in range(n_aliens):
                if i != j:
                    overlap = len(set(assigns[i]).intersection(assigns[j]))
                    overlaps.append(overlap)
        # add mean overlap
        overlap_means.append(np.mean(overlaps))

    total_avg_overlap = np.mean(overlap_means)
    total_avg_std = np.std(overlap_means)

    return total_avg_overlap, total_avg_std

def all_monte_carlos(total_n_att=48): # plus 1! so 0 att will be zero, 48 att is saved at monte_carlo[48]
    monte_carlos_mean = [0]*(total_n_att + 1)
    monte_carlos_std = [0]*(total_n_att + 1)
    for i in range(6, total_n_att + 1):
        mean, std = monte_carlo(i)
        monte_carlos_mean[i] = mean
        monte_carlos_std[i] = std
        print(i, ' ', monte_carlos_mean[i], monte_carlos_std[i])
    
    return monte_carlos_mean, monte_carlos_std

# ran once and now not necessary (saved in json)
mcs_m, mcs_std = all_monte_carlos()

with open('monte_carlos_means.txt', 'w') as f:
    f.write(json.dumps(mcs_m))

with open('monte_carlos_stds.txt', 'w') as g:
    g.write(json.dumps(mcs_std))