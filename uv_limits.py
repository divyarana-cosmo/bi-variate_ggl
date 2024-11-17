import numpy as np

def compute_transformed_limits_with_signs(cov_matrix, x_min, x_max, y_min, y_max):
    """
    Given a covariance matrix and limits in the x and y space, compute the transformed limits
    in the u and v space based on the eigenvector transformation and sign handling.

    Parameters:
        cov_matrix (ndarray): 2x2 covariance matrix.
        x_min (float): Minimum x value.
        x_max (float): Maximum x value.
        y_min (float): Minimum y value.
        y_max (float): Maximum y value.

    Returns:
        (u_min, u_max, v_min, v_max): Transformed limits in the u and v space.
    """
    # Step 1: Eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 2: Extract the eigenvectors (columns of eigenvectors matrix)
    v1 = eigenvectors[:, 0]  # first eigenvector (corresponds to lambda_1)
    v2 = eigenvectors[:, 1]  # second eigenvector (corresponds to lambda_2)

    # Step 3: Handle the signs of the eigenvectors
    # The signs of the components of the eigenvectors will help us determine the new limits
    u_min, u_max = float('inf'), float('-inf')
    v_min, v_max = float('inf'), float('-inf')

    # For u-axis (v1), the limits will be a combination of x and y based on the sign of v1 components
    u_min = min(v1[0] * x_min + v1[1] * y_min, v1[0] * x_max + v1[1] * y_max)
    u_max = max(v1[0] * x_min + v1[1] * y_min, v1[0] * x_max + v1[1] * y_max)

    # For v-axis (v2), the limits will be a combination of x and y based on the sign of v2 components
    v_min = min(v2[0] * x_min + v2[1] * y_min, v2[0] * x_max + v2[1] * y_max)
    v_max = max(v2[0] * x_min + v2[1] * y_min, v2[0] * x_max + v2[1] * y_max)

    # Adjust limits based on signs
    if v1[0] < 0:
        u_min, u_max = u_max, u_min
    if v2[0] < 0:
        v_min, v_max = v_max, v_min

    return u_min, u_max, v_min, v_max

# Example usage:
# Define the covariance matrix Sigma
Sigma = np.array([[2, 1], [1, 2]])

# Define the limits in x and y
x_min, x_max = 0, 1
y_min, y_max = 0, 1

# Compute the transformed limits in u and v
u_min, u_max, v_min, v_max = compute_transformed_limits_with_signs(Sigma, x_min, x_max, y_min, y_max)

# Print the transformed limits
print(f"Transformed limits for u: ({u_min:.2f}, {u_max:.2f})")
print(f"Transformed limits for v: ({v_min:.2f}, {v_max:.2f})")

