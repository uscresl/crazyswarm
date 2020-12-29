#!/usr/bin/env python

from copy import deepcopy
import  networkx as nx
import numpy as np
from multiprocessing import Process, Queue
import os
from pycrazyswarm import *

from kf_utils.DKFNode import DKFNode
from kf_utils.DKFNetwork import DKFNetwork
from kf_utils.target import Target, DEFAULT_H

from opt_utils.optimization import agent_opt
from opt_utils.formation import generate_coords


np.random.seed(42)

"""
Create the targets
"""
# TODO: how to associate these intitial positions with the actual drones?
t1 = Target(init_state=np.array([[10], [10], [1.], [1.]]))
t2 = Target(init_state=np.array([[-10], [10], [1.], [1.]]))
targets = [t1, t2]


"""
Create the trackers
"""
# TODO: how to associate these intitial positions with the actual drones?
num_trackers = 5
tracker_0_init_pos = np.array([0, 0, 20])
init_inter_tracker_dist = 15
fov = 30

node_attrs = {}

for n in range(num_trackers):
    pos = tracker_0_init_pos + np.array([n*init_inter_tracker_dist, 0, 0])
    region = [(pos[0] - fov, pos[0] + fov),
              (pos[1] - fov, pos[1] + fov)]

    node_attrs[n] = DKFNode(n,
                            [deepcopy(t) for t in targets],
                            position=pos,
                            region=region)


"""
Create inputs for the trackers
"""
# TODO:



"""
Create the network graph for the trackers
"""
# initialize in straight line
G = nx.Graph()
for i in range(num_trackers - 1):
    G.add_edge(i, i + 1)

weight_attrs = {}
for i in range(num_trackers):
    weight_attrs[i] = {}
    self_degree = G.degree(i)
    metropolis_weights = []
    for n in G.neighbors(i):
        degree = G.degree(n)
        mw = 1 / (1 + max(self_degree, degree))
        weight_attrs[i][n] = mw
        metropolis_weights.append(mw)
    weight_attrs[i][i] = 1 - sum(metropolis_weights)

network = DKFNetwork(node_attrs,
                     weight_attrs,
                     G,
                     targets)


"""
Create fixed failure sequence
"""
# set random sequence of drones to experience failure
num_failures = 7
failure_nodes = np.random.randint(5, size=num_failures)

# generate random matrices to add to R matrix of failed drone
r_mat_size = DEFAULT_H.shape[0]

rand_matrices = []
for _ in range(len(failure_nodes)):
    r = np.random.rand(r_mat_size, r_mat_size)
    rpd = np.dot(r, r.T)
    rand_matrices.append(rpd)



def p1():
    print('Current pid: {}'.format(os.getpid()))

    #cf.cmdVelocityWorld()

def p2(que):
    global network
    print('Current pid: {}'.format(os.getpid()))

    # TODO: when to apply failure
    failed_node = failure_nodes[0]
    fail_mat = rand_matrices[0]

    # TODO: read vicon and set actual positions
    # set target positions from vicon
    t1.state = np.array([[10], [10], [1.], [1.]])
    t2.state = np.array([[-10], [10], [1.], [1.]])

    # set tracker positions from vicon
    nodes = nx.get_node_attributes(network.network, 'node')
    for n in range(num_trackers):
        n.update_position(np.array([0, 0, 0]))

    # simulate local KF
    skip_config_generation = False
    for id, n in nodes.items():
        n.predict(len(nodes))
        ms = n.get_measurements([t1, t2])
        n.update(ms)

        # set flag if one of the trackers cannot see target
        if n.missed_observation:
            skip_config_generation = True

    # initialize consensus
    for id, n in nodes.items():
        n.init_consensus()

    # TODO: pass covariances, positions, Rs, flag, and fail information to queue
    for id, n in nodes.items():
        que.put(n.omega)
        que.put(n.position)
        que.put(n.R)

    que.put(skip_config_generation)
    que.put(failed_node)

    # push q2
    #que.put()


def p3(que):
    print('Current pid: {}'.format(os.getpid()))

    global network
    nodes = nx.get_node_attributes(network.network, 'node')
    current_weights = nx.get_node_attributes(network.network, 'weights')

    # read from queue
    covariance_data = []
    positions = []
    Rs = []
    for _ in range(len(nodes)):
        c = que.get()
        trace_c = np.trace(c)
        covariance_data.append(trace_c)
        positions.append(que.get())
        Rs.append(que.get())
    skip_config_generation = que.get()
    failed_node = que.get()

    if skip_config_generation:
        # do formation synthesis step only
        coords = generate_coords(network.network.adjacency_matrix(),
                                 positions, fov, Rs)
    else:
        # do optimization
        new_config, new_weights = agent_opt(network.network.adjacency_matrix(),
                                            current_weights,
                                            covariance_data,
                                            failed_node)
        # do formation synthesis
        coords = generate_coords(new_config,
                                 positions, fov, Rs)
        nx.set_node_attributes(network.network, new_weights, 'weights')


    # TODO: send coords
    #   send goTo


def main():
    qCmd = Queue()
    q1 = Queue()
    process1=Process(target=p1,args=(,))
    process2=Process(target=p2,args=(q2,))
    process3=Process(target=p3,args=(q2,))
    process1.start()
    process2.start()
    process3.start()

main()