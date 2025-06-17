import time
import sys

class timer:
    def __init__(self, op, wait_seconds):
        self.op = op
        self.wait_seconds = wait_seconds

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *exc_info):
        self.stop_time = time.time()
        self.elapsed_seconds = self.stop_time - self.start_time
        # if self.elapsed_seconds > self.wait_seconds:
        #     print(f"Op: '{self.op}' took: {round(self.elapsed_seconds, 2)} seconds.", file=sys.stderr)
