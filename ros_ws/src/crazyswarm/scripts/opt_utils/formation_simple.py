from copy import deepcopy
import numpy as np
import platform


def generate_coords(new_config, current_coords,
                    bbox=np.array([(-50, 50), (-50, 50), (10, 100)]),
                    delta=10, safe_dist=10, connect_dist=25, k=-0.1,
                    steps=1000, lax=True):
    """
    Uses Simulated Annealing to generate new coordinates given new
    network configuration.  This is used only for the two-step method.
    :param new_config: ndarray, matrix representing the new network configuration
    :param current_coords: dictionary, current robot positions
    :param bbox: region bounding box
    :param delta: delta is max movement
    :param safe_dist: safe distances between nodes
    :param connect_dist: connect distances between nodes
    :param steps: simulated annealing steps
    :param lax: boolean, whether or not to accept new coordinates even if some
        safety requirements are not met
    :return:
    """

    if platform.system() == 'Linux':
        invalid_iters_limit = 10
        steps = 10000
    else:
        invalid_iters_limit = 5

    # Set the temperature schedule for simulated annealing procedure
    H = np.logspace(1, 3, steps)
    temperature = np.logspace(1, -8, steps)

    new_coords = current_coords
    valid_config = False
    invalid_configs = 0
    while not valid_config:

        # Until appropriate cooling temperature is reached
        for i in range(steps):
            T = temperature[i]

            # Propose a new set of coordinates
            propose_coords = propose(new_coords, delta)

            # Compare the Energy function of the original coordinates and proposed coordinates
            current_E = energyCoverage(new_config, new_coords,
                                       H[i], k, safe_dist, connect_dist, bbox)
            propose_E = energyCoverage(new_config, propose_coords,
                                       H[i], k, safe_dist, connect_dist, bbox)

            # Accept the proposed coordinates if Energy is smaller
            # OR randomly accept based on temperature
            if propose_E < current_E:
                new_coords = deepcopy(propose_coords)
            else:
                p_accept = np.exp((-1 * (propose_E - current_E)) / T)
                accept_criteria = np.random.uniform(0, 1)
                if accept_criteria < p_accept:
                    new_coords = deepcopy(propose_coords)
            del propose_coords

        # After cooling temperature is reached, validate found coordinates
        valid_config = isValidConfig(new_coords, connect_dist, bbox)

        # Restart simulated annealing search if no valid_config is found
        if not valid_config:
            invalid_configs = invalid_configs + 1

        # If after sevaral searches fails,
        # return previous coordinates if lax = True,
        # otherwise return False
        if invalid_configs > invalid_iters_limit:
            print('could not find valid config')
            if lax:
                return new_coords
            else:
                return False

    return new_coords


def propose(current_coords, delta):
    """
    Propose New Coordinates
    :param current_coords: dictionary of nodes and positions
    :param delta: delta is max movement
    :return: dictionary of proposed coordinates for nodes
    """
    propose_coords = deepcopy(current_coords)
    node = np.random.choice(list(current_coords.keys()))
    dir = np.random.choice(3)
    d = np.random.uniform(-1 * delta, delta)

    old_pos = current_coords[node][dir]
    propose_coords[node][dir] = old_pos + d

    return propose_coords


def energyCoverage(config, propose_coords,
                   H, k, safe_dist, connect_dist, bbox):
    """
    Get Energy function
    :param config: ndarray, matrix representing the new network configuration
    :param propose_coords: the proposed robot coordinates
    :param H: entropy factor for simulated annealing
    :param k: some scaling factor for simulated annealing
    :param safe_dist: safe distances between nodes
    :param connect_dist: connect distances between nodes
    :param bbox: region bounding box
    :return: Energy value
    """
    n = len(propose_coords)
    total_distance_diffs = 0
    for i in range(n):
        for j in range(i, n):
            distance_diff = (np.linalg.norm(propose_coords[i] -
                                           propose_coords[j]) - D[i, j]) ** 2
            total_distance_diffs += distance_diff
    sum_box = 0
    sum_safe = 0
    sum_conn = 0

    n = len(propose_coords)
    for i in range(n):
        pos = propose_coords[i]
        sum_box_node = 0

        for d in range(len(bbox)):
            sum_box_node = sum_box_node + (ph(pos[d] - bbox[d, 1], H) +
                                           ph(bbox[d, 0] - pos[d], H))
        sum_box = sum_box + sum_box_node

        for j in range(i + 1, n):
            d = np.linalg.norm(propose_coords[i] - propose_coords[j])

            if config[i, j] > 0:
                sum_safe = sum_safe + ph(safe_dist - d, H)
            else:
                sum_conn = sum_conn + (ph(connect_dist - d, H))

    energy = (k * total_distance_diffs) + sum_box + sum_safe + sum_conn
    return energy


def isValidConfig(coords, safe_dist, bbox):
    """
    Checks if proposed coordinates are valid
    :param coords: dictionary or coordinates of nodes
    :param safe_dist: safe distances between nodes
    :param bbox: region bounding box
    :return:
    """
    n = len(coords)

    for i in range(n):
        # Check Position is within BBox
        x, y, z = coords[i]
        if not (bbox[0, 0] <= x <= bbox[0, 1]):
            print("x pos not in bbox")
            return False
        if not (bbox[1, 0] <= y <= bbox[1, 1]):
            print("y pos not in bbox")
            return False
        if not (bbox[2, 0] <= z <= bbox[2, 1]):
            print("z pos not in bbox")
            return False

        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if not (safe_dist <= d):
                print('too close to neighbor, {i}-{j}'.format(i=i, j=j))
                return False
    return True


def ph(x, H):
    """
    Penalty function for Simulated Annealing
    """
    if x < 0:
        return 0
    else:
        return np.exp(H * x)