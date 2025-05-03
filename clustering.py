import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.cluster import KMeans, k_means
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
# Load the data

def error (cluster_true, cluster_pred):
    error = 0
    total_pairs = 0
    for i in range(cluster_true.shape[0]):
        for j in range(i+1, cluster_true.shape[0]):
            if cluster_true[i] == cluster_true[j]:
                total_pairs += 1
                if cluster_pred[i] != cluster_pred[j]:
                    error += 1
    return error, total_pairs

data = fetch_covtype()


X = data.data
y = data.target

#randomly select 10000 samples
np.random.seed(0)
indices = np.random.choice(X.shape[0], 10000)
X_samples = X[indices]
y_samples = y[indices]

# Standardize the data

scaler = StandardScaler()
X_samples = scaler.fit_transform(X_samples)


# Reduce the dimensionality 

pca = PCA(0.92)
X_pca = pca.fit(X_samples)
X_pca = pca.transform(X_samples)
X_train = X_pca





# Perform KMeans clustering
kmeans = KMeans(n_clusters=7, random_state=0, n_init=10, init='k-means++', max_iter=300, tol=0.0001)

kmeans.fit(X_train)
y_kmeans = kmeans.predict(X_train)


error_kmeans, total_pairs = error(y_samples, y_kmeans)
print("total pairs  ", total_pairs)
print("error count of kmeans: ", error_kmeans)
#print("Accuracy: ", 1 - error_kmeans/total_pairs)



from sklearn.mixture import GaussianMixture
pca = PCA(0.98)
pca.fit(X_samples)
X_pca = pca.transform(X_samples)
X_train = X_pca

#best parameters from grid search {'gmm__covariance_type': 'full', 'gmm__init_params': 'random', 'gmm__max_iter': 300, 'gmm__n_components': 7, 'gmm__n_init': 10, 'gmm__random_state': 0, 'gmm__reg_covar': 1e-06, 'gmm__tol': 0.001, 'pca__n_components': 0.9800000000000001}

gmm = GaussianMixture(n_components=7, covariance_type='full', random_state=0, n_init=10, init_params='random', max_iter=300, tol=0.001, reg_covar=1e-6)
gmm.fit(X_train)
y_gmm = gmm.predict(X_train)

error_gmm, total_pairs = error(y_samples, y_gmm)
print("total pairs  ", total_pairs)
print("error count of gmm: ", error_gmm)


#run random baseline
y_random = np.random.randint(0, 7, y_samples.shape)

#error rate for random baseline
error_random, total_pairs_random = error(y_samples, y_random)
print(error_random, total_pairs_random)
print("total pairs  ", total_pairs_random)
print("Error count for random baseline: ",error_random)

