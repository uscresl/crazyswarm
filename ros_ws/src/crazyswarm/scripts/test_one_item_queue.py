from multiprocessing import Process
import queue
import time

import numpy as np

from one_item_queue import OneItemQueue


def producer(q, cycle_time, items):
    for i in range(items):
        q.put(i)
        if cycle_time > 0:
            time.sleep(cycle_time)
    q.put("END")


def consumer(q, cycle_time):
    items = []
    while True:
        if cycle_time > 0:
            time.sleep(cycle_time)
        try:
            x = q.get_nowait()
        except queue.Empty:
            continue
            # In real code, instead of repeating loop, could do other work.
        items.append(x)
        if x == "END":
            break
    #print(items)
    for i in range(len(items) - 2):
        assert items[i] < items[i + 1]
    assert items[-1] == "END"


def one_test(seed, with_sleep=False):
    np.random.seed(seed)
    if with_sleep:
        producer_cycle = np.random.uniform(0.001, 0.1)
        consumer_cycle = 10.0 ** np.random.uniform(-1.0, 1.0) * producer_cycle
        items = np.random.randint(0, 10)
    else:
        producer_cycle = 0.0
        consumer_cycle = 0.0
        items = np.random.randint(0, 1000)

    q = OneItemQueue()
    pp = Process(target=producer, args=(q, producer_cycle, items), name=f"P_{seed}")
    pc = Process(target=consumer, args=(q, consumer_cycle), name=f"C_{seed}")

    t0 = time.time()
    pp.start()
    pc.start()
    pc.join()
    t1 = time.time()

    if with_sleep:
        dur = t1 - t0
        min_total = (producer_cycle) * items
        assert dur >= min_total
        # generous fudge factor for spawn/join overhead and sleep() potentially
        # taking longer than requested.
        assert dur <= min_total + consumer_cycle + 1.0


def test_OneItemQueue():
    # The multiprocessing here is only to make the test run faster.
    # Can't use multiprocessing.Pool.map because pool processes can't spawn
    # their own child processes.
    N = 300
    seeds = list(range(N))
    procs = [Process(target=one_test, args=(s, False)) for s in seeds]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0

    procs = [Process(target=one_test, args=(s, True)) for s in seeds]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0


if __name__ == "__main__":
    test_OneItemQueue()
