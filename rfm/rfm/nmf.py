import numpy as np

def random_initialization(A, rank):
    """
    Initialize matrices W and H randomly.

    Parameters:
    - A: Input matrix
    - rank: Rank of the factorization

    Returns:
    - W: Initialized W matrix
    - H: Initialized H matrix
    """
    num_docs = A.shape[0]
    num_terms = A.shape[1]
    W = np.random.uniform(1, 2, (num_docs, rank))
    H = np.random.uniform(1, 2, (rank, num_terms))
    return W, H

def nndsvd_initialization(A, rank):
    """
    Initialize matrices W and H using Non-negative Double Singular Value Decomposition (NNDSVD).

    Parameters:
    - A: Input matrix
    - rank: Rank of the factorization

    Returns:
    - W: Initialized W matrix
    - H: Initialized H matrix
    """
    u, s, v = np.linalg.svd(A, full_matrices=False)
    v = v.T
    w = np.zeros((A.shape[0], rank))
    h = np.zeros((rank, A.shape[1]))

    w[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = np.sqrt(s[0]) * np.abs(v[:, 0].T)

    for i in range(1, rank):
        ui = u[:, i]
        vi = v[:, i]
        ui_pos = (ui >= 0) * ui
        ui_neg = (ui < 0) * -ui
        vi_pos = (vi >= 0) * vi
        vi_neg = (vi < 0) * -vi

        ui_pos_norm = np.linalg.norm(ui_pos, 2)
        ui_neg_norm = np.linalg.norm(ui_neg, 2)
        vi_pos_norm = np.linalg.norm(vi_pos, 2)
        vi_neg_norm = np.linalg.norm(vi_neg, 2)

        norm_pos = ui_pos_norm * vi_pos_norm
        norm_neg = ui_neg_norm * vi_neg_norm

        if norm_pos >= norm_neg:
            w[:, i] = np.sqrt(s[i] * norm_pos) / ui_pos_norm * ui_pos
            h[i, :] = np.sqrt(s[i] * norm_pos) / vi_pos_norm * vi_pos.T
        else:
            w[:, i] = np.sqrt(s[i] * norm_neg) / ui_neg_norm * ui_neg
            h[i, :] = np.sqrt(s[i] * norm_neg) / vi_neg_norm * vi_neg.T

    return w, h

def multiplicative_update(A, k, max_iter, init_mode='random'):
    """
    Perform Multiplicative Update (MU) algorithm for Non-negative Matrix Factorization (NMF).

    Parameters:
    - A: Input matrix
    - k: Rank of the factorization
    - max_iter: Maximum number of iterations
    - init_mode: Initialization mode ('random' or 'nndsvd')

    Returns:
    - W: Factorized matrix W
    - H: Factorized matrix H
    - norms: List of Frobenius norms at each iteration
    """
    if init_mode == 'random':
        W, H = random_initialization(A, k)
    elif init_mode == 'nndsvd':
        W, H = nndsvd_initialization(A, k)

    norms = []
    epsilon = 1.0e-10
    for _ in range(max_iter):
        # Update H
        W_TA = W.T @ A
        W_TWH = W.T @ W @ H + epsilon
        H *= W_TA / W_TWH

        # Update W
        AH_T = A @ H.T
        WHH_T = W @ H @ H.T + epsilon
        W *= AH_T / WHH_T

        norm = np.linalg.norm(A - W @ H, 'fro')
        norms.append(norm)

    return W, H, norms
