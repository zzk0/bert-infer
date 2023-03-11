import argparse
import contextlib
import time


def build_parser(name):
    parser = argparse.ArgumentParser(description="Command line interface for {}".format(name))
    parser.add_argument("--max_len", type=int, default=128, help="The maximum length of input sequence")
    parser.add_argument("--times", type=int, default=1000, help="The measure times")
    return parser


@contextlib.contextmanager
def timer(message: str = ""):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(
            "{} - Elapsed time: {:.4f} ms".format(
                message, (end_time - start_time) * 1000
            )
        )
