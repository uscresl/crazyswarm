from multiprocessing import Lock, Pipe, Semaphore
import pickle
import queue # only for queue.Empty exception


class OneItemQueue:
    """An queue for real-time streaming, i.e. old values are discarded.

    If the producer put()s two values in a row before the consumer calls get(),
    the first value is discarded forever.

    Only has a non-blocking get() because we expect the producer to be faster
    than the consumer.
    """
    def __init__(self):
        self.lock = Lock()
        self.sem = Semaphore(0)
        self.read, self.write = Pipe(duplex=False)

    def put(self, obj):
        """Replaces the current item, if any, with `obj`."""
        msg = pickle.dumps(obj)
        with self.lock:
            if self._is_set():
                self.read.recv_bytes()
            self.write.send_bytes(msg)
            self._set()

    def get_nowait(self):
        """Removes the current item and returns it, or raises queue.Empty."""
        msg = None
        with self.lock:
            if self._is_set():
                msg = self.read.recv_bytes()
                self._clear()
        if msg is not None:
            return pickle.loads(msg)
        raise queue.Empty

    # These methods are analogous to multiprocessing.Event, but Event has its
    # own lock, which appears to hurt performance significantly.
    def _is_set(self):
        if self.sem.acquire(False):
            self.sem.release()
            return True
        return False

    def _set(self):
        self.sem.acquire(False)
        self.sem.release()

    def _clear(self):
        self.sem.acquire(False)
