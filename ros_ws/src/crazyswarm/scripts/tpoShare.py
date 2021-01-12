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
t1 = Target(init_state=np.array([[10], [10], [1.], [1.]]))
t2 = Target(init_state=np.array([[-10], [10], [1.], [1.]]))
targets = [t1, t2]


"""
Create the trackers
"""
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



def p1(update_queue, state_queue):
    """

    :param update_queue: dictionary of form {'coords': {drone_id: drone_pos, ...}
                                             'new_config': adjacency_matrix,
                                             'new_weights: {drone_id: {neighbor_id: weight, ...} ...}
                                             }
        where drone_id is int
        adjacency_matrix is ndarray of shape (n, n)
        neighbor_id is int
        weight is float

    :param state_queue: dictionary of form {'coords': {drone_id: drone_pos, ...}
                                             'current_config': adjacency_matrix,
                                             'current_weights: {drone_id: {neighbor_id: weight, ...} ...}
                                             }
        where drone_id is int
        adjacency_matrix is ndarray of shape (n, n)
        neighbor_id is int
        weight is float

    :param network:
    :return:
    """
    print('Current pid: {}'.format(os.getpid()))


    #Set up for the crazyswarm
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    allcfs.takeoff(targetHeight=1.0, duration=1.0+1.0)
    timeHelper.sleep(1.5)

    byIdDict = allcfs.crazyfliesById

    while True:
        # Get latest optimization information
        if not update_queue.empty():
            update_cmd = update_queue.get()
            if update_cmd == "STOP":
                break
            else:
                for drone_id, drone_pos in update_cmd.items():
                    byIdDict[drone_id].goTo(drone_pos, 0, 1.0)
        
        new_state = {}
        for droneId in byIdDict.keys():
            new_state[droneId]= byIdDict[droneId].position()
        state_queue.put(new_state)
        timeHelper.sleepForRate(30)

def p2(state_queue, opt_queue, network, failure_nodes, rand_matrices):
    """

    :param state_queue: dictionary of form {'coords': {drone_id: drone_pos, ...}
                                             'current_config': adjacency_matrix,
                                             'current_weights: {drone_id: {neighbor_id: weight, ...} ...}
                                             }
        where drone_id is int
        adjacency_matrix is ndarray of shape (n, n)
        neighbor_id is int
        weight is float

    :param opt_queue: dictionary of form {drone_id : {'cov': drone_cov, 'pos': drone_pos, 'r': drone_r}, ...
                                          'skip_flag': skip_flag,
                                          'failed_drone': drone_id}
        where drone_id is int
        drone_cov is ndarray of shape (4,4)
        drone_pos is ndarray of shape (3,1)
        drone_r is ndarray of shape (4, 4)
        skip_flag is a boolean

    :param network: drone network object
    :param failure_nodes: list of nodes to apply sensor failures to throughout the simulation
    :param rand_matrices: the noise matrix to add to the failed node
    :return:
    """
    print('Current pid: {}'.format(os.getpid()))

    count_failures = 0
    while True:
        # Get latest positions from position_queue
        state = state_queue.get()
        positions = state['positions']
        current_config = state['current_config']
        current_weights = state['current_weights']

        nodes = nx.get_node_attributes(network.network, 'node')
        G = nx.from_numpy_matrix(current_config)
        network.network = G
        nx.set_node_attributes(network.network, current_weights, 'weights')
        nx.set_node_attributes(network.network, nodes, 'node')

        # randomly apply failure 50% of the time
        if np.random.rand() > 0.5:
            failed_node = failure_nodes[count_failures]
            fail_mat = rand_matrices[count_failures]
            nodes[failed_node].R += fail_mat
            count_failures += 1

        # set current tracker positions from data in queue

        for id, n in nodes.items():
            n.update_position(positions[id])

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

        opt_info = {}
        for id, n in nodes.items():
            opt_info[str(id)] = {'cov': n.omega,
                                 'pos': n.position,
                                 'r': n.R}
        opt_info['skip_flag'] = skip_config_generation
        opt_info['failed_drone'] = failed_node

        opt_queue.put(opt_info)

        if count_failures >= len(failure_nodes):
            break

    opt_queue.put('END')



def p3(opt_que, update_queue, network):
    """

    :param opt_queue: dictionary of form {drone_id : {'cov': drone_cov, 'pos': drone_pos, 'r': drone_r}, ...
                                        'skip_flag': skip_flag,
                                        'failed_drone': drone_id}
        where drone_id is int
        drone_cov is ndarray of shape (4,4)
        drone_pos is ndarray of shape (3,1)
        drone_r is ndarray of shape (4, 4)
        skip_flag is a boolean
    :param update_queue: dictionary of form {'coords': {drone_id: drone_pos, ...}
                                             'new_config': adjacency_matrix,
                                             'new_weights: {drone_id: {neighbor_id: weight, ...} ...}
                                             }
        where drone_id is int
        adjacency_matrix is ndarray of shape (n, n)
        neighbor_id is int
        weight is float
    :param network:
    :return:
    """
    print('Current pid: {}'.format(os.getpid()))

    while True:
        # Get latest optimization information
        opt_info = opt_que.get()

        if opt_info == 'END':
            break

        nodes = nx.get_node_attributes(network.network, 'node')
        current_weights = nx.get_node_attributes(network.network, 'weights')

        # read from queue
        covariance_data = []
        positions = []
        Rs = []
        for id, n in nodes.items():
            c = opt_info[str(id)]['cov']
            n.omega = c
            trace_c = np.trace(c)
            covariance_data.append(trace_c)

            n.update_position(opt_info[str(id)]['pos'])
            positions.append(opt_info[str(id)]['pos'])

            n.R = opt_info[str(id)]['r']
            Rs.append(opt_info[str(id)]['r'])

        skip_config_generation = opt_info['skip_flag']
        failed_node = opt_info['failed_drone']

        if skip_config_generation:
            # do formation synthesis step only
            coords = generate_coords(network.network.adjacency_matrix(),
                                     positions, fov, Rs)
            new_config = network.network.adjacency_matrix()
            new_weights = current_weights
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

        update = {'new_coords': coords,
                  'new_config': new_config,
                  'new_weights': new_weights}

        update_queue.put(update)

    update_queue.put('END')


def main():
    # TODO: instantiate global variables (network, failure_nodes, rand_matrices) in here
    qCmd = Queue()
    q1 = Queue()
    process1=Process(target=p1,args=())
    process2=Process(target=p2,args=(q2,))
    process3=Process(target=p3,args=(q2,))
    process1.start()
    process2.start()
    process3.start()

main()