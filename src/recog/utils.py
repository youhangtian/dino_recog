import sys
import os
import logging, logging.handlers

def get_logger(log_dir, log_file='log.txt'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    filename = os.path.join(log_dir, log_file)
    log_format = '%(asctime)s %(message)s'

    logger = logging.getLogger(log_file.split('.')[0])
    logger.setLevel(level=logging.INFO)

    #file_handler = logging.FileHandler(filename)
    file_handler = logging.handlers.RotatingFileHandler(filename, maxBytes=100*1024*1024, backupCount=9)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)    

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(stream_handler)

    return logger


class AverageMeter():
    def __init__(self, len=100):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

        self.len = len
        self.arr = []

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1

        if self.len > 0:
            self.arr.append(val)
            if len(self.arr) > self.len:
                self.sum -= self.arr[0]
                self.arr.pop(0)
                self.count -= 1

        self.avg = self.sum / self.count
        