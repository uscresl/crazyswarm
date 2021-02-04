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

from one_item_queue import OneItemQueue

np.random.seed(42)

Z = 1.0
sleepRate = 30
RADII = np.array([0.125, 0.125, 0.375])
NUM_TRACKERS = 6
NUM_TARGETS = 1
NUM_FAILURES = 3

BOUNDING_BOX_WIDTH = 3  # in x/y directions
TARGET_HEIGHT = 0.5
TRACKER_MIN_HEIGHT = TARGET_HEIGHT + RADII[2]
TRACKER_MAX_HEIGHT = 2.25

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


def p1(swarm, update_queue, state_queue, target_ids, tracker_id_map):
    """

    :param update_queue: dictionary of form {'coords': {drone_id: drone_pos, ...}}
        where drone_id is int

    :param state_queue: dictionary of form {'coords': {drone_id: drone_pos, ...}}
        where drone_id is int

    :param target_ids: ids of the targets (non-trackers)
    :param tracker_id_map: map of crazyflie ids to network ids
    :return:
    """
    print('Current pid: {}'.format(os.getpid()))

    #Set up for the crazyswarm
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    byIdDict = allcfs.crazyfliesById
    inv_tracker_id_map = {v: k for k, v in tracker_id_map.items()}

    idx = 0
    #enable collision avoidance
    for cf in allcfs.crazyflies:
        restCrazyflies = allcfs.crazyflies[:idx] + allcfs.crazyflies[(idx + 1):]
        cf.enableCollisionAvoidance(restCrazyflies, RADII)
        idx+=1
    #show ellipsoid    
    timeHelper.visualizer.showEllipsoids(0.95 * RADII)


    # for trackers in allcfs.crazyflies[:6]:
    for tracker_id in list(tracker_id_map.keys()):
        tracker = byIdDict[tracker_id]
        tracker.takeoff(targetHeight=TRACKER_MIN_HEIGHT, duration=1.0+Z)
        tracker.goTo(tracker.initialPosition + np.array([-1, -2, 2]), 0, 2)
    
    # for nonTrackers in allcfs.crazyflies[6:]:
    for non_tracker_id in target_ids:
        nonTracker = byIdDict[non_tracker_id]
        nonTracker.takeoff(targetHeight=TARGET_HEIGHT, duration=1.0+Z)

    timeHelper.sleep(1+Z)

    startTime = timeHelper.time()

    while True:
        #non-tracker go circle
        for target in target_ids:
            goCircle(timeHelper, byIdDict[target], startTime)

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
                    cf_id = inv_tracker_id_map[drone_id]
                    byIdDict[cf_id].goTo(drone_pos, 0, 5.0)
        
        new_state = {}
        for droneId in byIdDict.keys():
            if droneId in target_ids:
                network_id = droneId
            else:
                network_id = tracker_id_map[droneId]
            new_state[network_id]= byIdDict[droneId].position()

        state_queue.put({'coords': new_state})

        timeHelper.sleepForRate(sleepRate)

    state_queue.put('END')


def p2(state_queue, weights_queue, opt_queue, network, target_ids):
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
    :param target_ids: ids of the targets (non-trackers)
    :return:
    """
    print('Current pid: {}'.format(os.getpid()))


    # count_failures = 0
    while True:
        # Get latest positions from position_queue
        # state = state_queue.get()
        try:
            state = state_queue.get_nowait()
        except:
            continue

        if state == 'END':
            break

        positions = state['coords']
        # print("p2 while loop")
        try:
            weights = weights_queue.get(block=False)
            current_config = weights['new_config']
            current_weights = weights['new_weights']
        except:
            # print("using existing config and weights")
            current_config = network.adjacency_matrix()
            current_weights = nx.get_node_attributes(network.network, 'weights')

        nodes = nx.get_node_attributes(network.network, 'node')
        G = nx.from_numpy_matrix(current_config)
        network.network = G
        nx.set_node_attributes(network.network, current_weights, 'weights')
        nx.set_node_attributes(network.network, nodes, 'node')

        for i in range(len(target_ids)):
            t_id = target_ids[i]
            target_pos = positions[t_id]
            network.targets[i].state = \
                np.array([[target_pos[0]], [target_pos[1]], [1], [1]])

        # set current tracker positions from data in queue
        for id, n in nodes.items():
            n.update_position(positions[id])

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
        # opt_info['failed_drone'] = failed_node

        # send target positions (to fix bounding box for formation synthesis step)
        for t_id in target_ids:
            target_pos = positions[t_id]
            opt_info[str(t_id)] = {'pos': target_pos}

        opt_queue.put(opt_info)



def p3(opt_que, update_queue, weights_queue, network,
       failure_nodes, rand_matrices, target_ids):
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
    :param network: network object
    :param target_ids: ids of the targets (non-trackers)
    :return:
    """
    print('Current pid: {}'.format(os.getpid()))
    num_nodes = len(nx.get_node_attributes(network.network, 'node'))
    fov = {}
    for n in range(num_nodes):
        fov[n] = 5

    count_failures = 0
    while True:
        # Get latest optimization information
        # print("p3 waiting for opt info")
        try:
            opt_info = opt_que.get_nowait()
        except:
            continue
        print("p3 got opt info")

        nodes = nx.get_node_attributes(network.network, 'node')
        current_weights = nx.get_node_attributes(network.network, 'weights')

        # read from queue
        covariance_data = []
        positions = {}
        Rs = []

        # read target positions (to fix bounding box for formation synthesis step)
        target_x = []
        target_y = []
        for t_id in target_ids:
            t_pos = opt_info[str(t_id)]['pos']
            target_x.append(t_pos[0])
            target_y.append(t_pos[1])

        mean_x = np.mean(target_x)
        mean_y = np.mean(target_y)
        min_x = mean_x - BOUNDING_BOX_WIDTH
        max_x = mean_x + BOUNDING_BOX_WIDTH
        min_y = mean_y - BOUNDING_BOX_WIDTH
        max_y = mean_y + BOUNDING_BOX_WIDTH
        print((min_x, max_x), (min_y, max_y))

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

        # randomly apply failure some % of the time
        if np.random.rand() > 0.5:
            failed_node = failure_nodes[count_failures]
            fail_mat = rand_matrices[count_failures]
            nodes[failed_node].R += fail_mat
            count_failures += 1
        else:
            failed_node = None

        # failed_node = opt_info['failed_drone']
        if failed_node is None:
            skip_config_generation = True

        if skip_config_generation:
            # do formation synthesis step only
            print("running formation synthesis only")
            coords, _ = generate_coords(network.adjacency_matrix(),
                                     positions, fov, Rs,
                                        # bbox=np.array(
                                        #     [(-5, 5), (-5, 5), (1.5, 5)]),
                                        # bbox=np.array(
                                        #     [(-2.5, 2.5), (-4, 2), (1.5, 5)]),
                                        bbox=np.array([(min_x, max_x), (min_y, max_y),
                                                       (TRACKER_MIN_HEIGHT, TRACKER_MAX_HEIGHT)]),
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
                                        # bbox=np.array(
                                        #     [(-5, 5), (-5, 5), (1.5, 5)]),
                                        # bbox=np.array(
                                        #     [(-2.5, 2.5), (-4, 2), (1.5, 5)]),
                                        bbox=np.array([(min_x, max_x), (min_y, max_y),
                                                       (TRACKER_MIN_HEIGHT, TRACKER_MAX_HEIGHT)]),
                                        delta=3, safe_dist=1, connect_dist=2)
            nx.set_node_attributes(network.network, new_weights, 'weights')

        print("p3 sending coords")
        send_coords = {}
        for id, c in coords.items():
            send_coords[id] = c
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

        if count_failures >= len(failure_nodes):
            break

    print("p3 sending END command")
    update_queue.put('END')


def main():

    swarm = Crazyswarm()
    allcfs = swarm.allcfs
    byIdDict = allcfs.crazyfliesById

    y_pos_dict = {cfid: cf.position()[1] for cfid, cf in byIdDict.items()}
    y_pos_dict = dict(sorted(y_pos_dict.items(), key=lambda item: item[1]))
    ids_ordered_by_y_pos = list(y_pos_dict.keys())

    num_trackers = NUM_TRACKERS
    num_targets = NUM_TARGETS

    """
    Create the targets
    """

    target_ids = []
    targets = []
    for i in range(num_targets):
        target_id = ids_ordered_by_y_pos[i]
        target_ids.append(target_id)
        target_init_pos = byIdDict[target_id].position()

        t = Target(init_state=np.array([[target_init_pos[0]],
                                        [target_init_pos[1]],
                                        [1.], [1.]]))
        targets.append(t)


    """
    Create the trackers
    """
    fov = 5

    tracker_id_map = {}
    node_attrs = {}
    for n in range(num_trackers):
        tracker_id = ids_ordered_by_y_pos[num_targets+n]
        tracker_id_map[tracker_id] = n

        pos = byIdDict[tracker_id].position()
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
    num_failures = NUM_FAILURES
    failure_nodes = np.random.randint(num_trackers, size=num_failures)

    # generate random matrices to add to R matrix of failed drone
    r_mat_size = DEFAULT_H.shape[0]

    rand_matrices = []
    for _ in range(len(failure_nodes)):
        r = np.random.rand(r_mat_size, r_mat_size)
        rpd = np.dot(r, r.T)
        rand_matrices.append(rpd)


    state_queue = OneItemQueue()  # p1 to p2
    update_queue = Queue()  # p3 to p1
    weights_queue = Queue()  # p3 to p2
    opt_queue = OneItemQueue()  # p2 to p3

    process2=Process(target=p2,args=(state_queue, weights_queue, opt_queue,
                                     network, target_ids))
    process3=Process(target=p3,args=(opt_queue, update_queue, weights_queue,
                                     network, failure_nodes, rand_matrices,
                                     target_ids))
    process2.start()
    process3.start()

    p1(swarm, update_queue, state_queue, target_ids, tracker_id_map)

main()
