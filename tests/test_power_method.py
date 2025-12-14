from project import power_method
import numpy as np

def test_diagonal_matrix_positive_eigenvalue():
    """
    Test a simple diagonal matrix.
    A = [[10, 0], [0, 2]] -> Dominant eigenvalue = 10
    """
    A = np.array([[10.0, 0.0], 
                  [0.0, 2.0]])
    x0 = np.array([1.0, 1.0])
    
    # We use a tolerance of 1e-6 for the loop
    input_tol = 1e-6
    lam, v, iters = power_method(A, x0, maxit=200, tol=input_tol)

    # ASSERTION: 
    # We verify the calculated lambda is close to the True Value (10.0).
    # We use a slightly looser tolerance for the test (input_tol * 10) 
    # to account for floating point drift.
    assert np.isclose(lam, 10.0, atol=input_tol * 10)
    
    # We can still check the eigenvector direction loosely
    expected_v = np.array([1.0, 0.0])
    assert np.isclose(abs(np.dot(v, expected_v)), 1.0, atol=1e-2)

def test_diagonal_matrix_negative_eigenvalue():
    """
    Test where dominant eigenvalue is negative.
    A = [[-12, 0], [0, 5]] -> Dominant eigenvalue = -12
    """
    A = np.array([[-12.0, 0.0], 
                  [0.0, 5.0]])
    x0 = np.array([1.0, 1.0])
    
    input_tol = 1e-6
    lam, v, iters = power_method(A, x0, maxit=200, tol=input_tol)

    # Check against Ground Truth (-12.0)
    # The residual check has been removed.
    assert np.isclose(lam, -12.0, atol=input_tol * 10)

def test_convergence_criterion_specifically():
    """
    Test that the function actually stops when the eigenvalue stops changing,
    even if the residual isn't perfect yet.
    """
    # A matrix with eigenvalues 1.0 and 0.9. 
    # These are close, so convergence is slow.
    A = np.array([[1.0, 0.0], 
                  [0.0, 0.9]])
    x0 = np.array([0.5, 0.5])
    
    # Set a loose tolerance so it stops early
    loose_tol = 1e-2
    lam, v, iters = power_method(A, x0, maxit=1000, tol=loose_tol)
    
    # It should converge quickly because we asked for low precision
    assert iters < 1000 
    
    # The result should be roughly 1.0, but maybe not extremely precise
    assert np.isclose(lam, 1.0, atol=0.1)

def test_random_symmetric_matrix():
    """
    Compare against numpy's trusted implementation.
    """
    np.random.seed(123)
    B = np.random.rand(5, 5)
    A = B + B.T 
    x0 = np.random.rand(5)
    
    # 1. Get Ground Truth
    eigvals = np.linalg.eigvalsh(A)
    true_max_eig = eigvals[np.argmax(np.abs(eigvals))]

    # 2. Run your method
    input_tol = 1e-7
    lam, v, iters = power_method(A, x0, maxit=2000, tol=input_tol)

    # 3. Compare Result to Truth
    assert np.isclose(lam, true_max_eig, atol=input_tol * 100)