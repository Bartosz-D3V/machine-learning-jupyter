import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fmin_tnc

from recommender_system.array_to_vector import array_to_vector
from recommender_system.cofi_cost_function import cofi_cost_function
from recommender_system.load_mat import load_mat
from recommender_system.normalize_ratings import normalize_ratings
from recommender_system.visualize_matrix import visualize_matrix

r, y = load_mat('./data/ex8_movies.mat', 'R', 'Y')
theta, x, num_features, num_movies, num_users =\
    load_mat('./data/ex8_movie_params', 'Theta', 'X', 'num_features', 'num_movies', 'num_users')

print(f"Average rating for movie 1 (Toy Story) is: {round(float(np.mean(y[1, r[1, :] == 1])), 2)}")
visualize_matrix(plt, y)
plt.show()

folded_params = np.concatenate([x.ravel(order='F'), theta.ravel(order='F')])
cost_j, grad = cofi_cost_function(folded_params, y, r, int(num_users), int(num_movies), int(num_features), 0)
movies = np.genfromtxt('./data/movies.csv', delimiter='|', dtype=bytes).astype(str)
num_movies = np.size(movies, 0)
my_ratings = np.zeros((num_movies, 1))

# Toy Story - mark as 5 star movie
my_ratings[0] = 5
# Seven (Se7en)
my_ratings[10] = 4
# Braveheart
my_ratings[21] = 2
# Apollo 13
my_ratings[26] = 5
# Pulp Fiction
my_ratings[55] = 5
# Ace Ventura: Pet Detective
my_ratings[66] = 1
# Flipper
my_ratings[111] = 2
# Apocalypse Now
my_ratings[179] = 4

# Add my ratings to the y vector
y = np.append(my_ratings, y, axis=1)
r = np.append((my_ratings != 0).astype(int), r, axis=1)
y_norm, y_mean = normalize_ratings(y, r)
num_movies = np.size(y, 0)
num_users = np.size(y, 1)
num_features = 10

# Random parameters initialization
x = np.random.rand(num_movies, num_features)
theta = np.random.rand(num_users, num_features)
initial_parameters = np.concatenate([x.ravel(order='F'), theta.ravel(order='F')])
lambda_param = 10

# Minimize cost function
calculated_params = fmin_tnc(func=cofi_cost_function, x0=initial_parameters, args=(y_norm, r, num_users, num_movies, num_features, lambda_param))[0]
calculated_params = array_to_vector(calculated_params)
calculated_x = np.array(calculated_params[0:num_movies * num_features]).reshape((num_movies, num_features), order='F')
calculated_theta = np.array(calculated_params[num_movies * num_features:, :]).reshape((num_users, num_features), order='F')
prediction = calculated_x @ calculated_theta.T
my_predictions = np.asmatrix(prediction[:, 0]).T + y_mean
top_my_predictions_indices = np.argpartition(my_predictions, -5, axis=0)[-5:]
top_movies = movies[top_my_predictions_indices.T[0]][0][:,1]
print("")
