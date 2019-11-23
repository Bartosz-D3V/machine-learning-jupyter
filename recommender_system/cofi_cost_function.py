import numpy as np

from recommender_system.array_to_vector import array_to_vector


def cofi_cost_function(params, y, r, num_users, num_movies, num_features, lambda_param):
    params = array_to_vector(params)
    # Unroll parameters x & theta
    x = np.array(params[0:num_movies * num_features]).reshape((num_movies, num_features), order='F')
    theta = np.array(params[num_movies * num_features:, :]).reshape((num_users, num_features), order='F')

    regularized = ((lambda_param / 2) * np.sum(np.sum(theta ** 2))) + (lambda_param / 2) * np.sum(np.sum(x ** 2))
    cost_j = .5 * np.sum(np.sum(((x @ theta.T - y) * r) ** 2)) + regularized
    x_grad = ((x @ theta.T - y) * r) @ theta + lambda_param * x
    theta_grad = ((x @ theta.T - y) * r).T @ x + lambda_param * theta
    grad = array_to_vector(np.concatenate([x_grad.ravel(order='F'), theta_grad.ravel(order='F')]))
    return cost_j, grad
