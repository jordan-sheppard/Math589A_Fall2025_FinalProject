import numpy as np
import logging 

LOGGING_LEVEL = logging.INFO
logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================================================
# 1. Power method for dominant eigenpair
# =========================================================

def power_method(A: np.ndarray, x0: np.ndarray, maxit: int, tol: float):
    """Approximate the dominant eigenvalue and eigenvector of a real symmetric matrix A.

    Parameters
    ----------
    A : (n, n) ndarray
        Real symmetric matrix.
    x0 : (n,) ndarray
        Initial guess for eigenvector (nonzero).
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence in relative change of eigenvalue.

    Returns
    -------
    lam : float
        Approximate dominant eigenvalue.
    v : (n,) ndarray
        Approximate unit eigenvector (||v||_2 = 1).
    iters : int
        Number of iterations performed.
    """
    lam_0 = x0 @ A @ x0
    iters = 0
    while iters < maxit:
        iters += 1
        # Power iteration
        y = A @ x0
        x0 = y / np.linalg.norm(y, ord=2)
        lam = x0 @ A @ x0

        # Convergence check in relative error
        err = abs(lam - lam_0) / abs(lam)
        logging.debug(f"iters = {iters}, lam_k = {lam}, err = {err}")
        if err < tol:
            break

        # Set up for next iteration 
        lam_0 = lam

    if iters == maxit:
        logging.error(f"Convergence not achieved after {maxit} iterations.")
    return lam, x0, iters



# =========================================================
# 2. Rank-k image compression via SVD
# =========================================================
def svd_compress(image: np.ndarray, k: int):
    """Compute a rank-k approximation of a grayscale image using SVD.

    Parameters
    ----------
    image : (m, n) ndarray
        Grayscale image matrix.
    k : int
        Target rank (1 <= k <= min(m, n)).

    Returns
    -------
    image_k : (m, n) ndarray
        Rank-k approximation of the image.
    rel_error : float
        Relative Frobenius error ||image - image_k||_F / ||image||_F.
    compression_ratio : float
        (Number of stored parameters in image_k) / (m * n).
    """
    m, n = image.shape
    max_rank = min(m,n)
    if k <= 0 or k > max_rank:
        raise ValueError(f"Provided rank out of bounds. Should be in [1, max_rank] = [1, {max_rank}]")
    
    U, S, Vh = np.linalg.svd(image, full_matrices=False)
    image_k = (U[:,:k] * S[:k]) @ Vh[:k,:]
    rel_error = np.linalg.norm(image - image_k, ord='fro') / np.linalg.norm(image, ord='fro')
    compression_ratio = k * (m + n + 1) / (m * n)
    
    return image_k, rel_error, compression_ratio


# =========================================================
# 3. SVD-based feature extraction
# =========================================================

def svd_features(image, p):
    """Extract SVD-based features from a grayscale image.

    Parameters
    ----------
    image : (m, n) ndarray
        Grayscale image matrix.
    p : int
        Number of leading singular values to use (p <= min(m, n)).

    Returns
    -------
    feat : (p + 2,) ndarray
        Feature vector consisting of:
        [normalized sigma_1, ..., normalized sigma_p, r_0.9, r_0.95]
    """
    singular_values = np.linalg.svd(image, compute_uv = False)
    energies = np.cumsum(singular_values) / np.sum(singular_values)
    r_0_9 = np.argmax(energies >= 0.9) + 1
    r_0_95 = np.argmax(energies >= 0.95) + 1
    return np.hstack((energies[:p], [r_0_9, r_0_95]))



# =========================================================
# 4. Two-class LDA: training
# =========================================================

def lda_train(X, y):
    """Train a two-class LDA classifier.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix (rows = samples, columns = features).
    y : (N,) ndarray
        Labels, each 0 or 1.

    Returns
    -------
    w : (d,) ndarray
        Discriminant direction vector (not necessarily unit length).
    threshold : float
        Threshold in 1D projected space for classifying 0 vs 1.
    """
    # Separate out two training classes
    X_A = X[y == 0, :]
    X_B = X[y == 1, :]

    # Find class means
    mu_A = np.mean(X_A, axis=0)     # Should be shape (d,) array
    mu_B = np.mean(X_B, axis=0)     # Should be shape (d,) array
    X_A_centered = X_A - mu_A       # Should be shape (N_A, d) array
    X_B_centered = X_B - mu_B       # Should be shape (N_B, d) array

    # Create within-class scatter matrix
    S_W = (X_A_centered.T @ X_A_centered) + (X_B_centered.T @ X_B_centered) # Should be shape (d,d) array

    # Create discriminant direction vector
    LAMBDA = 1e-6       # Regularization parameter 
    S_W_reg = S_W + LAMBDA * np.eye(S_W.shape[0])       # Should be shape (d,d) array
    w = np.linalg.solve(S_W_reg, mu_B - mu_A)           # Should be shape (d,) array

    # Find threshold value 
    m_A = np.mean(X_A @ w)
    m_B = np.mean(X_B @ w)
    threshold = (m_A + m_B) / 2.

    return w, threshold

# =========================================================
# 5. Two-class LDA: prediction
# =========================================================

def lda_predict(X, w, threshold):
    """Predict class labels using a trained LDA classifier.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix.
    w : (d,) ndarray
        Discriminant direction (from lda_train).
    threshold : float
        Threshold (from lda_train).

    Returns
    -------
    y_pred : (N,) ndarray
        Predicted labels (0 or 1).
    """
    projections = X @ w 
    in_B = projections >= threshold
    return 1 * in_B     # 1 if in B, 0 if in A


# =========================================================
# Simple self-test on the example data
# =========================================================

def _example_run():
    """Run a tiny end-to-end test on the example dataset, if available.

    This function is for local testing only and will NOT be called by the autograder.
    """
    try:
        data = np.load("project_data_example.npz")
    except OSError:
        print("No example data file 'project_data_example.npz' found.")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Sanity check shapes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    p = min(5, min(X_train.shape[1], X_train.shape[2]))
    print(f"Using p = {p} leading singular values for features.")

    # Build feature matrices
    def build_features(X):
        feats = []
        for img in X:
            feats.append(svd_features(img, p))
        return np.vstack(feats)

    try:
        Xf_train = build_features(X_train)
        Xf_test = build_features(X_test)
    except NotImplementedError:
        print("Implement 'svd_features' first to run this example.")
        return

    print("Feature dimension:", Xf_train.shape[1])

    try:
        w, threshold = lda_train(Xf_train, y_train)
    except NotImplementedError:
        print("Implement 'lda_train' first to run this example.")
        return

    try:
        y_pred = lda_predict(Xf_test, w, threshold)
    except NotImplementedError:
        print("Implement 'lda_predict' first to run this example.")
        return

    accuracy = np.mean(y_pred == y_test)
    print(f"Example test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    # This allows students to run a quick local smoke test.
    _example_run()
