from sys import implementation
from tracemalloc import start
from turtle import st
import numpy as np
from hmmlearn import hmm

rewards = np.loadtxt('rewards.txt')   
rewards = rewards.astype(int)         


# hmmlearn expects observations in a 2D array [n_samples, n_features].
# Here we have a single discrete feature per time step, so we reshape:


num_categories = 3  # {0, 1, 2}
N = len(rewards)
#reshaping observations
observations = np.zeros((N, num_categories), dtype=int)
for i, obs in enumerate(rewards):
    observations[i, obs] = 1

    


n_states = 9
n_rewards = 3
#tested with a uniform startprob as a prior but did not affect the results
#startprob = [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
#startprob = np.array(startprob)


# Fit model with EM algorithm without true transition matrix
model = hmm.MultinomialHMM(n_components=n_states, n_iter=1000, tol=1e-6, random_state=0, params='ste', init_params='ste')
model.n_features = n_rewards
model.startprob_ = np.random.dirichlet(np.ones(n_states))
model.fit(observations)

print("Start Probabilities without true transition matrix:", model.startprob_)
print("Transition Matrix without true transtion matrix:", model.transmat_)
print("Emission Probabilities without true transition matrix:", model.emissionprob_)
trained_transmat = model.transmat_


# Code task 15 true transition matrix
true_transmat = np.zeros((n_states, n_states))
for i in range(3):
    for j in range(3):
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j))
        if i < 2:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < 2:
            neighbors.append((i, j + 1))

        current_state = i * 3 + j
        for neighbor in neighbors:
            neighbor_state = neighbor[0] * 3 + neighbor[1]
            true_transmat[current_state, neighbor_state] = 1 / len(neighbors)

# Initialize a new HMM with true transition matrix and set initial parameters to se and params to se
model_with_true_transmat = hmm.MultinomialHMM(n_components=n_states, n_iter=1000, tol=1e-6, random_state=0, init_params='se', params='se')
model_with_true_transmat.n_features = n_rewards
model_with_true_transmat.startprob_ = np.random.dirichlet(np.ones(n_states))
model_with_true_transmat.emissionprob_ = np.random.dirichlet(np.ones(n_rewards), size=n_states)
model_with_true_transmat.transmat_ = true_transmat



# Train the model 
model_with_true_transmat.fit(observations)

# Extract learned parameters with true transition matrix
learned_startprob_true_transmat = model_with_true_transmat.startprob_
learned_emissionprob_true_transmat = model_with_true_transmat.emissionprob_

print("Learned Start Probabilities with True Transition Matrix:", learned_startprob_true_transmat)
print("Learned Emission Probabilities with True Transition Matrix:", learned_emissionprob_true_transmat)
print("True Transition Matrix:", true_transmat)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For a nicer plotting style

def plot_transition_matrix(transmat, title):
    """
    Plots the given transition matrix as a heatmap.
    
    Args:
        transmat (numpy.ndarray): The transition matrix to plot.
        title (str): Title of the heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(transmat, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("Next State")
    plt.ylabel("Current State")



plot_transition_matrix(trained_transmat, "Learned Transition Matrix with EM")

plot_transition_matrix(true_transmat, "True Transition Matrix")

plt.show()