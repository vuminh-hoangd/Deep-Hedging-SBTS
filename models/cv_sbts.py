import numpy as np


def kernel(x, h):
    return np.where(np.abs(x) < h, (h ** 2 - x ** 2) ** 2, 0.0)


def simulate_final_step_batch(Y_known, X_ref, weights_conditioning, N, N_pi, h, deltati, L):
    """
    Given a query path's known values Y_known[0..N-1], and precomputed
    conditioning weights (M,) reflecting the match against reference paths,
    simulate L independent realizations of the terminal value Y_N.
    """
    ref_target = X_ref[:, N]              # X_ref[:, i+1] with i = N-1
    Y_last = Y_known[N - 1]

    weights_tilde = weights_conditioning * np.exp((ref_target - Y_last) ** 2 / (2 * deltati))

    X_ = np.full(L, Y_last)
    v_time = np.linspace(0, deltati, N_pi + 1)
    Brownian = np.random.normal(0, 1, size=(L, N_pi))

    for k in range(N_pi):
        timeprev = v_time[k]
        timestep = v_time[k + 1] - v_time[k]

        if k == 0:
            expec_den = np.sum(weights_conditioning)
            expec_num = np.sum(weights_conditioning[None, :] * (ref_target[None, :] - X_[:, None]), axis=1)
        else:
            diff = ref_target[None, :] - X_[:, None]                       # (L, M)
            termtoadd = weights_tilde[None, :] * np.exp(-diff ** 2 / (2 * (deltati - timeprev)))
            expec_den = np.sum(termtoadd, axis=1)                          # (L,)
            expec_num = np.sum(termtoadd * diff, axis=1)                   # (L,)

        drift = np.where(expec_den > 0, (1.0 / (deltati - timeprev)) * (expec_num / np.where(expec_den > 0, expec_den, 1.0)), 0.0)
        X_ = X_ + drift * timestep + Brownian[:, k] * np.sqrt(timestep)

    return X_  # (L,) realizations of the terminal value


def conditioning_weights_plain(Y_known, X_ref, h, N):
    """Full-history conditioning (the 'plain' version): weights accumulate over i=1,...,N-1."""
    M = X_ref.shape[0]
    weights = np.full(M, 1.0 / M)
    for i in range(1, N):
        weights = weights * kernel(X_ref[:, i] - Y_known[i], h)
    return weights


def conditioning_weights_markov(Y_known, X_ref, h, K, N):
    """Order-K conditioning: only match the last K known points before the final transition."""
    M = X_ref.shape[0]
    weights = np.ones(M)
    for j in range(K):
        idx = N - K + j
        weights = weights * kernel(X_ref[:, idx] - Y_known[idx], h)
    return weights


def cv_mse_plain(X_train_ref, X_val_ref, h, N, N_pi, deltati, L, idx_q):
    """MSE_h for the plain version, using a FIXED set of query path indices (idx_q)
    so every candidate h is evaluated on exactly the same validation paths."""
    sq_errors = []
    for q in idx_q:
        Y_known = X_val_ref[q, :N]        # values at t_0,...,t_{N-1}
        true_terminal = X_val_ref[q, N]

        weights = conditioning_weights_plain(Y_known, X_train_ref, h, N)
        if weights.sum() == 0:
            continue  # bandwidth too small for this query, skip (will show up as poor h in aggregate)

        realizations = simulate_final_step_batch(Y_known, X_train_ref, weights, N, N_pi, h, deltati, L)
        pred_mean = realizations.mean()
        sq_errors.append((pred_mean - true_terminal) ** 2)

    return np.mean(sq_errors) if sq_errors else np.inf


def cv_mse_markov(X_train_ref, X_val_ref, h, K, N, N_pi, deltati, L, idx_q):
    """MSE_{h,k} for the Markovian version, using a FIXED set of query path indices."""
    sq_errors = []
    for q in idx_q:
        Y_known = X_val_ref[q, :N]
        true_terminal = X_val_ref[q, N]

        weights = conditioning_weights_markov(Y_known, X_train_ref, h, K, N)
        if weights.sum() == 0:
            continue

        realizations = simulate_final_step_batch(Y_known, X_train_ref, weights, N, N_pi, h, deltati, L)
        pred_mean = realizations.mean()
        sq_errors.append((pred_mean - true_terminal) ** 2)

    return np.mean(sq_errors) if sq_errors else np.inf