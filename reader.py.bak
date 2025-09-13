import os
from utils import xp

class SlidingComplex64Reader:
    """
    Sliding-window reader for complex64 samples stored in a binary file.
    Keeps a GPU-resident cache (CuPy array) of up to `capacity` elements.

    Access pattern assumptions:
      - Requests are non-decreasing in start index (monotonic forward),
      - Random access happens only within the most-recent window.

    File layout:
      - Raw interleaved complex64 (8 bytes/sample).
    """
    dtype = xp.complex64
    itemsize = 8  # complex64

    def __init__(self, file_path, capacity=1_000_000, prefetch_ratio=0.25):
        """
        :param file_path: path to raw complex64 file
        :param capacity: max number of complex64 samples to keep in cache
        :param prefetch_ratio: extra fraction of capacity to prefetch on reload (0..1)
        """
        self.file_path = file_path
        self.capacity = int(capacity)
        self.prefetch_extra = int(self.capacity * float(prefetch_ratio))
        self._f = open(file_path, 'rb', buffering=0)  # unbuffered to make seeks exact
        self.total_values = os.path.getsize(file_path) // self.itemsize

        # GPU cache state
        self.base = 0                 # global start index of cache window
        self.buf = xp.empty((0,), dtype=self.dtype)  # GPU buffer (view onto current window)
        self.valid = 0                # number of valid elements in buf

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    def _read_into_gpu(self, start, need_len):
        """
        (Re)load the window so that [start, start+need_len) fits,
        with optional right-side prefetch.
        """
        if start >= self.total_values:
            # nothing to read
            self.base = self.total_values
            self.buf = xp.empty((0,), dtype=self.dtype)
            self.valid = 0
            return

        # Choose how many to load: requested + prefetch, capped by file and capacity
        load_len = min(
            max(need_len, min(self.capacity, self.total_values - start)),
            self.capacity
        )
        # Add prefetch if capacity allows and file has more to the right
        extra = min(self.prefetch_extra, self.total_values - (start + load_len))
        load_len = min(self.capacity, load_len + extra)

        # Seek and read directly into GPU using cupy.fromfile
        self._f.seek(start * self.itemsize, os.SEEK_SET)
        data = xp.fromfile(self._f, dtype=self.dtype, count=load_len)

        # Update window
        self.base = start
        self.buf = data
        self.valid = int(data.size)

    def get(self, start, length):
        """
        Return a CuPy view of [start : start+length) as complex64 on GPU.
        If beyond EOF, returns a shorter view (possibly empty).
        """
        if length <= 0:
            return self.buf[:0]

        # Clamp to EOF
        start = int(start)
        end = min(start + int(length), self.total_values)
        if end <= start:
            return self.buf[:0]

        # If current window covers the request, return a view
        if self.valid > 0 and (start >= self.base) and (end <= self.base + self.valid):
            lo = start - self.base
            hi = end - self.base
            return self.buf[lo:hi]

        # Otherwise, advance the window. Strategy: place `start` at window left.
        need_len = end - start
        try:
            self._read_into_gpu(start, need_len)
        except:
            return None

        # Compute view again (may be shorter if near EOF)
        avail_end = min(self.base + self.valid, end)
        if avail_end <= start:
            return self.buf[:0]
        lo = start - self.base
        hi = avail_end - self.base
        return self.buf[lo:hi]

    def __del__(self):
        self.close()

