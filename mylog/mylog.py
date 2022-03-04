import os
from datetime import datetime
from time import time


class MyLog(object):
    def __init__(self, log_file='log/log_file', file_output=True, screen_output=True, reset=False):
        if not os.path.exists("log"):
            os.makedirs("log")
        self.st = time()
        self.log_file = log_file
        self.file_output = file_output
        self.screen_output = screen_output
        if reset:
            f = open(log_file, "w")
            f.close()

    def get_start(self):
        return self.st

    def set_start(self, st):
        self.st = st

    def reset(self):
        self.st = time()

    def get_time(self):
        return time() - self.st

    def set_file_output(self, file_output):
        self.file_output = file_output

    def set_screen_output(self, screen_output):
        self.screen_output = screen_output

    def set_log_file(self, log_file):
        self.log_file = log_file

    def log(self, msg, file_output=None, screen_output=None):
        if file_output is None:
            file_output = self.file_output
        if screen_output is None:
            screen_output = self.screen_output

        current_time = self.get_time()

        if file_output:
            f = open(self.log_file, "a+", encoding='utf-8')
            f.write(str(datetime.now()) + ' : ' + msg + '\n')
            f.close()

        if screen_output:
            print("%15.4f" % current_time, ':', msg)
