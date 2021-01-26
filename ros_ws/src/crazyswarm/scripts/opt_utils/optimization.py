from copy import deepcopy
import csv
import cvxopt as cvx
import math
import numpy as np
import os
import picos as pic
from picos.solvers import SolverError
import platform
from scipy.linalg import block_diag
from heapq import *
import itertools
counter = itertools.count()


def magnitude(x):
    return int(math.log10(abs(x)))



def agent_opt(adj_mat, current_weights, covariance_data, ne=1, failed_node=None):
    """
    Runs the agent optimization problem
    :param adj_mat: the Adjacency matrix
    :param current_weights: current node weights
    :param covariance_data: list of each node's covariance matrix
    :param ne: limit for number of edges to change
    :param failed_node: the node that fails
    :return: new adjacency matrix and new weights
    """
    edge_mod_limit = ne * 2
    n = adj_mat.shape[0]
    beta = 1 / n
    node_bin = np.zeros((1, n))
    if failed_node is not None:
        node_bin[0][failed_node] = 1

    # Reducing Magnitude if necessary
    covariance_data = np.nan_to_num(covariance_data)
    magnitude_covs = [magnitude(cov) for cov in covariance_data]
    if max(magnitude_covs) > 15:
        print('rescaling matrix magnitude, magnitude too high')
        covariance_data = [cov * (10 ** (-1 * max(magnitude_covs)))
                           for cov in covariance_data]

    # Init Problem
    problem = pic.Problem()

    # Add Variables
    A = problem.add_variable('A', adj_mat.shape,
                             'symmetric')  # A is an SDP var
    PI = problem.add_variable('PI', adj_mat.shape,
                              'binary')  # PI is a binary var
    mu = problem.add_variable('mu', 1)  # mu is an SDP var

    # Set Objective
    problem.set_objective('min',
                          -1 * node_bin * A * np.array(covariance_data).T)

    # Set Constraints
    problem.add_constraint(mu >= 0.001)
    problem.add_constraint(mu < 1)
    problem.add_constraint(
        (A * np.ones((n, 1))) == np.ones((n, 1)))  # Constraint 1
    problem.add_constraint((beta * np.dot(np.ones(n).T, np.ones(n))) +
                           (1 - mu) * np.eye(n) >= A)  # Constraint 2

    for i in range(n):
        problem.add_constraint(A[i, i] > 0)  # Constraint 6
        for j in range(n):
            if i == j:
                problem.add_constraint(PI[i, j] == 1.0)  # Constraint 3
            else:
                problem.add_constraint(A[i, j] > 0)  # Constraint 7
                problem.add_constraint(A[i, j] <= PI[i, j])  # Constraint 8

    problem.add_constraint(
        abs(PI - adj_mat) ** 2 <= edge_mod_limit)  # Constraint 9

    problem.solve(verbose=0, solver='mosek')
    problem_status = problem.status
    print('status: {s}'.format(s=problem_status))
    if problem_status not in ['integer optimal', 'optimal']:
        return adj_mat, current_weights

    new_config = np.zeros(adj_mat.shape)
    new_weights = {}
    for i in range(n):
        new_weights[i] = {}

    for i in range(n):
        nw = A[i, i].value
        if nw == 0:
            nw = 0.1
        new_weights[i][i] = nw
        new_config[i, i] = 1
        for j in range(i + 1, n):
            if round(PI[i, j].value) == 1:
                new_config[i, j] = round(PI[i, j].value)
                new_config[j, i] = round(PI[j, i].value)
                nw = A[i, j].value
                if nw == 0:
                    nw = 0.1
                new_weights[i][j] = nw
                new_weights[j][i] = nw
    print(new_config)
    return new_config, new_weights
