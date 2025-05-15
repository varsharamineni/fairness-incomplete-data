import numpy as np
import pandas as pd
import itertools

# Function to initialize parameters
def initialise_params(K, M, labels, seed=3):
    np.random.seed(seed)
    params = []

    for i in range(len(M)):
        t = np.random.random(M[i] * K).reshape(M[i], K)
        t /= t.sum(axis=0, keepdims=True)
        params.append(t)

    pi = np.random.random(K)
    pi /= pi.sum()
    params.append(pi)

    return params, labels


# Log-likelihood computation
def ll(df1, df2, params, labels, K):
    N1, N2 = len(df1), len(df2)
    ll1, ll2 = np.zeros(N1), np.zeros(N2)

    cols1, cols2 = df1.columns, df2.columns

    for n in range(N1):
        for k in range(K):
            m1 = 1
            for col in cols1:
                idx = np.where(labels == col)[0][0]
                m1 *= params[idx][df1[col][n], k]
            ll1[n] += m1 * params[-1][k]

    for n in range(N2):
        for k in range(K):
            m2 = 1
            for col in cols2:
                idx = np.where(labels == col)[0][0]
                m2 *= params[idx][df2[col][n], k]
            ll2[n] += m2 * params[-1][k]

    ll = np.log(ll1).sum() + np.log(ll2).sum()
    return ll


# E-step
def Estep(df1, df2, params, labels, K):
    N1, N2 = len(df1), len(df2)
    q1, q2 = np.zeros((N1, K)), np.zeros((N2, K))

    cols1, cols2 = df1.columns, df2.columns

    for n in range(N1):
        for k in range(K):
            m1 = 1
            for col in cols1:
                idx = np.where(labels == col)[0][0]
                m1 *= params[idx][df1[col][n], k]
            q1[n, k] = m1 * params[-1][k]
    q1 /= q1.sum(axis=1, keepdims=True)

    for n in range(N2):
        for k in range(K):
            m2 = 1
            for col in cols2:
                idx = np.where(labels == col)[0][0]
                m2 *= params[idx][df2[col][n], k]
            q2[n, k] = m2 * params[-1][k]
    q2 /= q2.sum(axis=1, keepdims=True)

    return q1, q2


# M-step
def Mstep(df1, df2, q1, q2, K, M, labels):
    N1, N2 = len(df1), len(df2)
    params = [np.zeros((M[i], K)) for i in range(len(M))]
    pi = np.zeros(K)
    params.append(pi)

    cols1, cols2 = df1.columns, df2.columns

    for idx, col in enumerate(labels):
        if col in cols1 and col not in cols2:
            for m in range(M[idx]):
                i = df1[df1[col] == m].index
                params[idx][m, :] = q1[i, :].sum(axis=0)
            params[idx] /= params[idx].sum(axis=0, keepdims=True)

        if col in cols2 and col not in cols1:
            for m in range(M[idx]):
                i = df2[df2[col] == m].index
                params[idx][m, :] = q2[i, :].sum(axis=0)
            params[idx] /= params[idx].sum(axis=0, keepdims=True)

        if col in cols1 and col in cols2:
            for m in range(M[idx]):
                i1 = df1[df1[col] == m].index
                i2 = df2[df2[col] == m].index
                params[idx][m, :] = q1[i1, :].sum(axis=0) + q2[i2, :].sum(axis=0)
            params[idx] /= params[idx].sum(axis=0, keepdims=True)

    for k in range(K):
        params[-1][k] = q1[:, k].sum() + q2[:, k].sum()
    params[-1] /= (N1 + N2)

    return params


# EM algorithm
def em_algo(df1, df2, M, labels, tol=1e-6, max_iter=500, K=3, seed=3):
    labels = np.array(labels)
    params, labels = initialise_params(K, M, labels, seed)
    log_likelihood = np.zeros(max_iter + 1)
    log_likelihood[0] = ll(df1, df2, params, labels, K)

    for i in range(max_iter):
        print(i)
        q1, q2 = Estep(df1, df2, params, labels, K)
        params = Mstep(df1, df2, q1, q2, K, M, labels)
        log_likelihood[i + 1] = ll(df1, df2, params, labels, K)

        if np.abs(log_likelihood[i + 1] - log_likelihood[i]) < tol:
            break
        print(f"Iteration {i + 1}: Log-likelihood = {log_likelihood[i + 1]:.4f}")

    log_likelihood = log_likelihood[log_likelihood != 0]
    return log_likelihood, params



def compute_joint_em(params, M, K):
    """
    params: list of final EM parameters (len = num_vars + 1), where
            params[i] = p(X_i | Z), shape = [M[i], K]
            params[-1] = pi (mixing proportions), shape = [K]
    M: list of number of categories for each variable
    K: number of latent components
    """
    num_vars = len(M)
    joint_dist = {}

    # All combinations of variable values
    all_combos = list(itertools.product(*[range(m) for m in M]))

    for x in all_combos:  # e.g., (0,1,0)
        prob = 0
        for k in range(K):
            p = params[-1][k]  # pi_k
            for i in range(num_vars):
                p *= params[i][x[i], k]  # p(X_i = x_i | Z = k)
            prob += p
        
        # Convert the combination to a string and format it as p000, p001, ...
        key = 'p' + ''.join(str(val) for val in x)
        joint_dist[key] = prob

    return joint_dist