import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import SparsePCA

def center_data(X):
    row_means = np.mean(X, axis=1, keepdims=True)
    Y = X - row_means
    return Y


def compute_covariance_matrix(Y):
    covariance_matrix = np.dot(Y.T, Y)
    return covariance_matrix


def compute_sparse_pca(Y, K, alpha=1):
    spca = SparsePCA(n_components=K-1, alpha=alpha, random_state=42)
    spca.fit(Y.T)
    components = spca.components_.T
    top_eigenvectors = np.dot(Y.T, components)
    explained_variance = np.var(np.dot(components.T, Y), axis=1)
    return explained_variance, top_eigenvectors


'''
# compute_sparse_pca replaces the two original functions to get the K-1 eigenvectors

def compute_eigenvalues_and_eigenvectors(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvalues, eigenvectors


def select_top_eigenvectors(eigenvalues, eigenvectors, K):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_indices = sorted_indices[:K-1]
    top_eigenvalues = eigenvalues[top_indices]
    top_eigenvectors = eigenvectors[:, top_indices]
    return top_eigenvalues, top_eigenvectors
'''


def compute_bounds(Y, top_eigenvalues):
    y_squared_sum = np.sum(np.sum(Y ** 2, axis=0))
    eigenvalues_sum = np.sum(top_eigenvalues)
    lower_bound = y_squared_sum - eigenvalues_sum 
    upper_bound = y_squared_sum
    return lower_bound, upper_bound


def compute_projection_matrix(top_eigenvectors):
    P = np.zeros((top_eigenvectors.shape[0], top_eigenvectors.shape[0]))
    for i in range(top_eigenvectors.shape[1]):
        v = top_eigenvectors[:, i].reshape(-1, 1)
        P += np.dot(v, v.T)
    return P

def compute_connectivity_matrix(P, alpha=0.5):
    row_sums = np.sqrt(np.diag(P))
    normalization = np.outer(row_sums, row_sums)
    R = P / normalization

    C = np.where(R >= alpha, P, 0)
    return C


def compute_laplacian(C):
    D = np.diag(np.sum(C, axis=1))
    L = D - C
    return L


def compute_fiedler_vector(L):
    _, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1]
    return fiedler_vector


def permute_connectivity_matrix(fiedler_vector):
    pi = np.argsort(fiedler_vector)
    return pi


def compute_cluster_crossing(C, pi, K):
    n = len(C)
    m = int(np.ceil(n/K))
    rho = np.zeros(n)
    for i in range(n):
        t = min(i+1, n - i, m)
        rho[i] = (m / t) * sum(C[pi[i - j], pi[i + j]] for j in range(1, t))
    return rho


def smooth_cluster_crossing(C, pi, rho, K):
    n = len(C)
    m = int(np.ceil(n/K))
    smoothed_rho = np.zeros_like(rho)

    for i in range(1, n-1):
        t = min(i + 1, n - i, m)

        if i < n - 1:
            rho_plus_half = 0
            t_counter = 0
            for j in range(1, t):
                if i + j + 1 < n:
                    rho_plus_half += C[pi[i - j], pi[i + j + 1]]
                    t_counter += 1
            rho_plus_half *= m/t_counter if t_counter > 0 else 0
        else:
            rho_plus_half = 0

        if i > 0:
            rho_minus_half = 0
            t_counter = 0
            for j in range(1, t):
                if i + j - 1 < n:
                    rho_minus_half += C[pi[i - j], pi[i + j - 1]]
                    t_counter += 1
            rho_minus_half *= m/t_counter if t_counter > 0 else 0
        else:
            rho_minus_half = 0

        smoothed_rho[i] = (rho_plus_half / 4) + (rho[i] / 2) + (rho_minus_half / 4)

    return smoothed_rho


def find_valleys(C, pi, rho, K):
    n = len(rho)
    neighborhood = 1  # Start with 1 neighbor on each side
    current_valleys = []

    while neighborhood <= n // 2:  # Avoid exceeding half the array length
        # Find valleys for the current neighborhood size
        candidate_valleys = []
        for i in range(neighborhood, n - neighborhood):
            if all(rho[i] <= rho[i - j] for j in range(1, neighborhood + 1)) and \
               all(rho[i] <= rho[i + j] for j in range(1, neighborhood + 1)):
                candidate_valleys.append(i)

        # Resolve ties for valleys in the same neighborhood
        unique_valleys = []
        i = 0
        while i < len(candidate_valleys):
            start_idx = i
            while i + 1 < len(candidate_valleys) and candidate_valleys[i + 1] - candidate_valleys[i] <= neighborhood:
                i += 1

            # If there are multiple valleys in the same neighborhood, resolve ties
            if i > start_idx:
                tied_valleys = candidate_valleys[start_idx:i + 1]
                # Find the most distinct valley based on neighbor differences
                best_valley = min(tied_valleys, key=lambda x: min(
                    C[pi[x], pi[x - 1]] if x - 1 >= 0 else float('inf'),
                    C[pi[x], pi[x + 1]] if x + 1 < n else float('inf')
                ))
                unique_valleys.append(best_valley)
            else:
                unique_valleys.append(candidate_valleys[i])

            i += 1

        candidate_valleys = unique_valleys

        # Check if the number of valleys matches K-1
        if len(candidate_valleys) == K - 1:
            return candidate_valleys  # Found the desired valleys

        # If we fall below K-1 valleys, revert to the previous step
        if len(candidate_valleys) < K - 1:
            # Select the K-1 valleys with the lowest values from the last valid neighborhood
            if len(current_valleys) > K - 1:
                current_valleys = sorted(current_valleys, key=lambda x: rho[x])[:K - 1]
            return current_valleys

        # Otherwise, update and expand the neighborhood
        current_valleys = candidate_valleys
        neighborhood += 1

    # Final fallback: If no valid K-1 valleys are found, use the closest approximation
    if len(current_valleys) > K - 1:
        current_valleys = sorted(current_valleys, key=lambda x: rho[x])[:K - 1]

    return current_valleys


def recursive_clustering(C, K):

    # Step 1: Compute Laplacian and Fiedler vector
    L = compute_laplacian(C)
    fiedler_vector = compute_fiedler_vector(L)

    # Step 2: Permute connectivity matrix based on Fiedler vector ordering
    pi = permute_connectivity_matrix(fiedler_vector)

    # Step 3: Compute cluster crossing
    rho = compute_cluster_crossing(C, pi, K)
    smoothed_rho = smooth_cluster_crossing(C, pi, rho, K)

    # Step 4: Locate valley points in the cluster crossing
    valleys = find_valleys(C, pi, smoothed_rho, K)
    #print(valleys)

    # Step 5: Initial cluster assignment based on valleys
    clusters = []
    start = 0

    for valley in valleys:
        # Assign the range before the valley to a cluster
        cluster_range = pi[start:valley]
        clusters.append(cluster_range)

        # Compare valley point's connectivity to left and right neighbors
        if valley - 1 >= 0:
            left_connectivity = C[pi[valley], pi[valley - 1]]
        else:
            left_connectivity = float('-inf')  # No left neighbor for the first valley

        if valley + 1 < len(C):
            right_connectivity = C[pi[valley], pi[valley + 1]]
        else:
            right_connectivity = float('-inf')  # No right neighbor for the last valley

        # Assign valley to the cluster with the higher connectivity
        if left_connectivity >= right_connectivity:
            clusters[-1] = np.append(clusters[-1], pi[valley])
            start = valley + 1
        else:
            start = valley
            continue  # Skip this valley as it belongs to the next cluster

    # Add the final range after the last valley
    clusters.append(pi[start:])

    return clusters, smoothed_rho


def get_optimal_value(clusters, Y):
    km = 0
    for indices in clusters:
        cluster_observations = Y[:, indices]
        mean_cluster = np.mean(cluster_observations, axis=1)
        sum_cluster = np.zeros_like(mean_cluster)
        for i in indices:
            column = Y[:, i]
            sum_cluster +=  (column - mean_cluster)**2
        km += sum_cluster.sum()
    return km


def run_clustering(X, K):
    
    Y = center_data(X)

    top_eigenvalues, top_eigenvectors = compute_sparse_pca(Y, K)

    P = compute_projection_matrix(top_eigenvectors)

    C = compute_connectivity_matrix(P)

    clusters, smoothed_rho = recursive_clustering(C, K)

    km = get_optimal_value(clusters, Y)

    return clusters, km, smoothed_rho


def get_clustering_accuracy(labels, preds):

    contingency_matrix = confusion_matrix(labels, preds)
    
    # Use Hungarian algorithm to find optimal label assignment
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    
    # Map predicted labels to true labels
    label_map = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_preds = np.array([label_map[label] for label in preds])
    
    # Compute accuracy
    accuracy = np.mean(mapped_preds == labels)
    return accuracy


def get_pred_labels(C):
    size = sum(len(x) for x in C)
    pred_labels = np.zeros(size)
    for i, array in enumerate(C):
        pred_labels[array] = i
    pred_labels = pred_labels.astype(int).tolist()
    return pred_labels
