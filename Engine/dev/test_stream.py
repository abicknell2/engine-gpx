# type: ignore
"test a threaded stream pong response"
import json
import logging
import math
import os
import queue
import threading
from threading import Thread
import time

from flask import Flask, Response

import utils.logger as logger

logging.basicConfig(level=logging.DEBUG)

PONG_INTERVAL = 3  # seconds

app = Flask(__name__)


def running_function(queue):
    "a function to run for awhile"
    logger.debug("starting the running function")
    res = []
    for i in range(10):
        res.append(math.sqrt(i))
        logger.debug(f"sqrt of {i} is {res[-1]}")
        time.sleep(1.0)
    # put the result in the queue
    queue.put(res)


@app.route("/ping")
def pingpong():
    pipeline = queue.Queue(maxsize=1)
    th = Thread(
        target=running_function,
        args=(pipeline, ),
    )  # include the extra ',' to make sure this is a tuple and not just a grouping
    th.start()

    logger.debug("getting response")

    def generator():
        logger.debug("starting the generator")
        last_update = time.time()
        while th.is_alive():
            now = time.time()
            if now - last_update > PONG_INTERVAL:
                logger.debug("issue a pong")
                # if greater than the interval give a pong
                last_update = now
                logger.debug("ponging the client")
                yield "pong"

        logger.debug("thread is done")
        # return the results
        yield json.dumps(pipeline.get())

    # th.start()

    return Response(generator())


@app.route("/test/stream")
def streamtest():

    def gen():
        for i in range(10):
            time.sleep(0.5)
            yield str(i)

        yield "done"

    return Response(gen())


def bad_process():
    "a long_running process that is bad"
    res = []
    for i in range(20):
        res.append(i)
        print(i)
        time.sleep(0.5)

    # after looping raise an error
    raise ValueError("this process has failed")


@app.route("/test/join")
def test_join():
    "test joining a thread"
    t = Thread(target=bad_process)
    t.start()

    def gen():

        t.join()  # this waits for the thread to finish
        while t.isAlive():
            time.sleep(1)
            yield "..."
        yield "done"

    return Response(gen())


def exec_handler(args):
    raise ValueError("threading exception")


@app.route("/test/hook")
def test_hook():
    t = Thread(target=bad_process)
    threading.excepthook = exec_handler

    def gen():

        t.start()  # this waits for the thread to finish
        while t.isAlive():
            time.sleep(1)
            yield "..."
        yield "done"

    return Response(gen())


def test_thread():
    th = Thread(target=running_function)
    th.start()
    interval = 0.5
    last_check = time.time()
    while th.is_alive():
        if time.time() - last_check > interval:
            last_check = time.time()
            print(th.is_alive())

    print("thread completed")
    # TODO:  get the results
    # kill the thread?


if __name__ == "__main__":
    # app.run(host='0.0.0.0', threaded=True)
    host = os.getenv("APP_HOST", "127.0.0.1")
    app.run(host=host)
    # test_thread()
