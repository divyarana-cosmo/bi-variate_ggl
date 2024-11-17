import numpy as np
from scipy.special import erf, sqrt

def gaussian_integral_with_error_function(a, b, c, x_limits, y_limits):
    """
    Compute the two-dimensional Gaussian integral with correlated variables
    by diagonalizing the covariance matrix and using error functions.

    Parameters:
    a, b: coefficients for x^2 and y^2
    c: the correlation term between x and y
    x_limits: tuple (x_min, x_max) for the x-integration limits
    y_limits: tuple (y_min, y_max) for the y-integration limits

    Returns:
    The result of the Gaussian integral
    """

    # Step 1: Calculate eigenvalues (lambda_1 and lambda_2) of the covariance matrix
    # Covariance matrix is: [[a, c], [c, b]]
    det = (a - b)**2 + 4 * c**2
    lambda_1 = (a + b + sqrt(det)) / 2
    lambda_2 = (a + b - sqrt(det)) / 2

    # Step 2: Calculate the eigenvectors (v1, v2) corresponding to the eigenvalues
    # Solve (Sigma - lambda I) v = 0 for each eigenvalue
    v_1 = np.array([b - lambda_1, c])  # Eigenvector for lambda_1
    v_2 = np.array([b - lambda_2, c])  # Eigenvector for lambda_2

    # Normalize the eigenvectors
    v_1 /= np.linalg.norm(v_1)
    v_2 /= np.linalg.norm(v_2)

    # Step 3: Construct the rotation matrix P
    P = np.column_stack([v_1, v_2])

    # Step 4: Compute the Jacobian determinant of the transformation
    jacobian_det = np.abs(np.linalg.det(P))  # Jacobian determinant is |det(P)|

    # Step 5: Use the error function for the Gaussian integral

    # Extract integration limits for x and y
    x_min, x_max = x_limits
    y_min, y_max = y_limits

    # Step 6: Compute the integral for the first transformed variable u
    # The integral for u (transformed axis corresponding to lambda_1)
    integral_u = (sqrt(np.pi) / 2) * (
        erf(np.sqrt(lambda_1) * x_max) - erf(np.sqrt(lambda_1) * x_min)
    )

    # Step 7: Compute the integral for the second transformed variable v
    # The integral for v (transformed axis corresponding to lambda_2)
    integral_v = (sqrt(np.pi) / 2) * (
        erf(np.sqrt(lambda_2) * y_max) - erf(np.sqrt(lambda_2) * y_min)
    )

    # Step 8: Combine the integrals and normalize by sqrt(lambda_1 * lambda_2)
    result = (integral_u * integral_v) / (jacobian_det * sqrt(lambda_1 * lambda_2))

    return result

# Example coefficients for the quadratic form
a = 1.0  # Coefficient for x^2
b = 1.0  # Coefficient for y^2
c = 0.5  # Correlation between x and y (c < min(a, b))

# Define the integration limits
x_limits = (0, 1)  # Integration limits for x
y_limits = (0, 1)  # Integration limits for y

# Compute the result of the Gaussian integral
integral_result = gaussian_integral_with_error_function(a, b, c, x_limits, y_limits)

# Print the result
print(f"The result of the two-dimensional Gaussian integral using error functions is: {integral_result:.6f}")

