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

Z = 1.0
sleepRate = 30
RADII = np.array([0.125, 0.125, 0.375])

def goCircle(timeHelper, cf, startTime, totalTime=4, radius=1, kPosition=1):
    startPos = cf.initialPosition + np.array([0, 0, Z])
    center_circle = startPos - np.array([radius, 0, 0])

    time = timeHelper.time() - startTime
    omega = 2 * np.pi / totalTime
    vx = -radius * omega * np.sin(omega * time)  
    vy = radius * omega * np.cos(omega * time)
    desiredPos = center_circle + radius * np.array(
        [np.cos(omega * time), np.sin(omega * time), 0])
    errorX = desiredPos - cf.position() 
    cf.cmdVelocityWorld(np.array([vx, vy, 0] + kPosition * errorX), yawRate=0)
    timeHelper.sleepForRate(sleepRate)


def p1(swarm, update_queue, state_queue):
    """

    :param update_queue: dictionary of form {'coords': {drone_id: drone_pos, ...}}
        where drone_id is int

    :param state_queue: dictionary of form {'coords': {drone_id: drone_pos, ...}}
        where drone_id is int

    :param network:
    :return:
    """
    print('Current pid: {}'.format(os.getpid()))

    #Set up for the crazyswarm
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    idx = 0
    #enable collision avoidance
    for cf in allcfs.crazyflies:
        restCrazyflies = allcfs.crazyflies[:idx] + allcfs.crazyflies[(idx + 1):]
        cf.enableCollisionAvoidance(restCrazyflies, RADII)
        idx+=1
    #show ellipsoid    
    timeHelper.visualizer.showEllipsoids(0.95 * RADII)

    #allcfs.takeoff(targetHeight=1.0, duration=1.0+Z)
    #timeHelper.sleep(1+Z)

    for trackers in allcfs.crazyflies[:6]:
        trackers.takeoff(targetHeight=2.0, duration=1.0+Z)
    
    for nonTrackers in allcfs.crazyflies[6:]:
        nonTrackers.takeoff(targetHeight=1.0, duration=1.0+Z)

    timeHelper.sleep(1+Z)

    byIdDict = allcfs.crazyfliesById

    startTime = timeHelper.time()

    while True:
        #non-tracker go circle
        goCircle(timeHelper, byIdDict[6], startTime)
        goCircle(timeHelper, byIdDict[7], startTime)

        # Get latest optimization information
        if not update_queue.empty():
            print("p1 getting latest update")
            update_cmd = update_queue.get()
            if update_cmd == "END":
                for cf in allcfs.crazyflies:
                    cf.goTo(cf.initialPosition + np.array([0, 0, Z]), 0, 5.0)
                timeHelper.sleep(10.0)
                allcfs.land(targetHeight=0.06, duration=2.0)
                timeHelper.sleep(2.0)
                break
            else:
                for drone_id, drone_pos in update_cmd["coords"].items():
                    byIdDict[drone_id].goTo(drone_pos, 0, 5.0)
        
        new_state = {}
        for droneId in byIdDict.keys():
            new_state[droneId]= byIdDict[droneId].position()
        state_queue.put({'coords': new_state})
        timeHelper.sleepForRate(sleepRate)


def p2(state_queue, weights_queue, opt_queue, network, failure_nodes, rand_matrices):
    """

    :param state_queue: dictionary of form {'coords': {drone_id: drone_pos, ...}}
        where drone_id is int

    :param weights_queue: dictionary of form {'new_config': adjacency_matrix,
                                             'new_weights: {drone_id: {neighbor_id: weight, ...} ...}
                                             }
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
        positions = state['coords']
        print("p2 while loop")
        try:
            weights = weights_queue.get(block=False)
            current_config = weights['new_config']
            current_weights = weights['new_weights']
        except:
            print("using existing config and weights")
            current_config = network.adjacency_matrix()
            current_weights = nx.get_node_attributes(network.network, 'weights')

        nodes = nx.get_node_attributes(network.network, 'node')
        G = nx.from_numpy_matrix(current_config)
        network.network = G
        nx.set_node_attributes(network.network, current_weights, 'weights')
        nx.set_node_attributes(network.network, nodes, 'node')

        # randomly apply failure 80% of the time
        if np.random.rand() > 0.5:
            failed_node = failure_nodes[count_failures]
            fail_mat = rand_matrices[count_failures]
            nodes[failed_node].R += fail_mat
            count_failures += 1
        else:
            failed_node = None

        # set current tracker positions from data in queue
        for id, n in nodes.items():
            n.update_position(positions[id+1])

        # simulate local KF
        skip_config_generation = False
        for id, n in nodes.items():
            n.predict(len(nodes))
            ms = n.get_measurements(network.targets)
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

    while not opt_queue.empty():
        pass
    print("sending END command")
    opt_queue.put('END')



def p3(opt_que, update_queue, weights_queue, network):
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
    num_nodes = len(nx.get_node_attributes(network.network, 'node'))
    fov = {}
    for n in range(num_nodes):
        fov[n] = 5

    while True:
        # Get latest optimization information
        opt_info = opt_que.get()
        print("p3 got opt info")

        if opt_info == 'END':
            break

        nodes = nx.get_node_attributes(network.network, 'node')
        current_weights = nx.get_node_attributes(network.network, 'weights')

        # read from queue
        covariance_data = []
        positions = {}
        Rs = []

        for id, n in nodes.items():
            c = opt_info[str(id)]['cov']
            n.omega = c
            trace_c = np.trace(c)
            covariance_data.append(trace_c)

            n.update_position(opt_info[str(id)]['pos'])
            positions[id] = opt_info[str(id)]['pos']

            n.R = opt_info[str(id)]['r']
            Rs.append(opt_info[str(id)]['r'])

        skip_config_generation = opt_info['skip_flag']
        failed_node = opt_info['failed_drone']
        if failed_node is None:
            skip_config_generation = True

        if skip_config_generation:
            # do formation synthesis step only
            coords, _ = generate_coords(network.adjacency_matrix(),
                                     positions, fov, Rs,
                                        bbox=np.array(
                                            [(-5, 5), (-5, 5), (1.5, 5)]),
                                        delta=3, safe_dist=1, connect_dist=2)
            new_config = network.adjacency_matrix()
            new_weights = current_weights
        else:
            # do optimization
            print("running optimization")
            print(failed_node)
            new_config, new_weights = agent_opt(network.adjacency_matrix(),
                                                current_weights,
                                                covariance_data,
                                                failed_node)
            # do formation synthesis
            coords, _ = generate_coords(new_config,
                                     positions, fov, Rs,
                                        bbox=np.array(
                                            [(-5, 5), (-5, 5), (1.5, 5)]),
                                        delta=3, safe_dist=1, connect_dist=2)
            nx.set_node_attributes(network.network, new_weights, 'weights')

        print("p3 sending coords")
        send_coords = {}
        for id, c in coords.items():
            send_coords[id+1] = c
        # coords = {1: np.array([0., 0., 0]),
        #           2: np.array([1., 0., 0]),
        #           3: np.array([2., 0., 0]),
        #           4: np.array([3., 0., 0])
        #           }
        print(coords)
        update = {'coords': send_coords}
        update_queue.put(update)

        weight_update = {'new_config': new_config,
                         'new_weights': new_weights}
        # weight_update = {'new_config': network.adjacency_matrix(),
        #                  'new_weights': current_weights}
        weights_queue.put(weight_update)

    update_queue.put('END')


def main():

    swarm = Crazyswarm()
    allcfs = swarm.allcfs
    byIdDict = allcfs.crazyfliesById

    t1_pos = byIdDict[6].position()
    t2_pos = byIdDict[7].position()

    """
    Create the targets
    """
    t1 = Target(init_state=np.array([[t1_pos[0]], [t1_pos[1]], [1.], [1.]]))
    t2 = Target(init_state=np.array([[t2_pos[0]], [t2_pos[2]], [1.], [1.]]))
    targets = [t1, t2]

    """
    Create the trackers
    """
    num_trackers = 5
    # tracker_0_init_pos = np.array([0, 0, 20])
    # init_inter_tracker_dist = 15
    # fov = 30
    fov = 5

    node_attrs = {}

    for n in range(num_trackers):
        # pos = tracker_0_init_pos + np.array(
        #     [n * init_inter_tracker_dist, 0, 0])
        pos = byIdDict[n+1].position()
        region = [(pos[0] - fov, pos[0] + fov),
                  (pos[1] - fov, pos[1] + fov)]

        node_attrs[n] = DKFNode(n,
                                [deepcopy(t) for t in targets],
                                position=pos,
                                region=region)


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

    state_queue = Queue()  # p1 to p2
    update_queue = Queue(1)  # p3 to p1
    weights_queue = Queue(1)  # p3 to p2
    opt_queue = Queue(1)  # p2 to p3

    process2=Process(target=p2,args=(state_queue, weights_queue, opt_queue,
                                     network, failure_nodes, rand_matrices))
    process3=Process(target=p3,args=(opt_queue, update_queue, weights_queue, network))
    process2.start()
    process3.start()

    p1(swarm, update_queue, state_queue)

main()