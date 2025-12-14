import pytest
import numpy as np
from project import svd_compress

@pytest.fixture
def sample_image():
    """Generates a random 50x40 grayscale image for testing."""
    np.random.seed(42)
    return np.random.rand(50, 40)


def test_output_shapes_and_types(sample_image):
    """
    Check that the function returns the correct types and shapes.
    """
    m, n = sample_image.shape
    k = 10
    
    img_k, rel_err, comp_ratio = svd_compress(sample_image, k)
    
    assert isinstance(img_k, np.ndarray)
    assert img_k.shape == (m, n)
    assert isinstance(rel_err, float)
    assert isinstance(comp_ratio, float)

def test_full_rank_reconstruction(sample_image):
    """
    If k = min(m, n), the reconstruction should be exact (within float limits),
    and relative error should be close to 0.
    """
    m, n = sample_image.shape
    k = min(m, n)
    
    img_k, rel_err, comp_ratio = svd_compress(sample_image, k)
    
    # Error should be negligible (~ machine epsilon)
    assert np.isclose(rel_err, 0.0, atol=1e-7)
    
    # Reconstructed image should match original
    assert np.allclose(img_k, sample_image, atol=1e-7)

def test_rank_of_output(sample_image):
    """
    Verify that the output matrix actually has rank <= k.
    """
    k = 5
    img_k, _, _ = svd_compress(sample_image, k)
    
    # Calculate rank using numpy
    actual_rank = np.linalg.matrix_rank(img_k)
    assert actual_rank <= k

def test_error_monotonicity(sample_image):
    """
    As k increases, the relative error should decrease (or stay same).
    k=1 should have high error, k=full should have 0 error.
    """
    k_values = [1, 5, 10, 20]
    errors = []
    
    for k in k_values:
        _, rel_err, _ = svd_compress(sample_image, k)
        errors.append(rel_err)
        
    # Check that errors are strictly descending (or equal)
    # err[0] > err[1] > ...
    assert np.all(np.diff(errors) <= 0)
    
    # Check bounds: Error should be between 0 and 1
    assert all(0.0 <= e <= 1.0 for e in errors)

def test_compression_ratio_calculation(sample_image):
    """
    Test that the compression ratio follows the standard formula:
    (k*m + k*n + k) / (m*n)
    where we store k vectors for U, k vectors for V, and k singular values.
    """
    m, n = sample_image.shape
    k = 5
    
    _, _, comp_ratio = svd_compress(sample_image, k)
    
    # Expected: (U_k size + Sigma_k size + V_k size) / Total Pixels
    # U_k is (m, k), Sigma_k is (k,), V_k is (k, n) -> technically (n, k) stored
    num_params = k * m + k + k * n
    expected_ratio = num_params / (m * n)
    
    assert np.isclose(comp_ratio, expected_ratio)

def test_diagonal_matrix_logic():
    """
    Test with a simple diagonal matrix where SVD is obvious.
    Image = diag(10, 5, 2, 0)
    """
    # 4x4 image
    image = np.diag([10.0, 5.0, 2.0, 0.0])
    
    # If we keep k=2, we should keep singular values 10 and 5, drop 2.
    # The reconstruction should be diag(10, 5, 0, 0)
    k = 2
    img_k, rel_err, _ = svd_compress(image, k)
    
    expected_img = np.diag([10.0, 5.0, 0.0, 0.0])
    
    assert np.allclose(img_k, expected_img)
    
    # Manual Frobenius norm calc
    # ||A||_F = sqrt(10^2 + 5^2 + 2^2) = sqrt(129)
    # ||A - Ak||_F = sqrt(2^2) = 2
    # Expected rel error = 2 / sqrt(129)
    expected_err = 2.0 / np.sqrt(129.0)
    assert np.isclose(rel_err, expected_err)

def test_invalid_k_input(sample_image):
    """
    Test that the function raises ValueError if k is out of bounds.
    """
    m, n = sample_image.shape
    
    # k = 0 is invalid (usually)
    with pytest.raises(ValueError):
        svd_compress(sample_image, 0)
        
    # k > min(m, n) is invalid
    with pytest.raises(ValueError):
        svd_compress(sample_image, min(m, n) + 1)