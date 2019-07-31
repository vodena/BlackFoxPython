from datetime import datetime
import sys


class CsvLogWriter(object):

    def __init__(self, log_file=sys.stdout, only_change=True):
        self.log_file = log_file
        self.write_string(
            'Time,Status,Generation,Total generations,Validation set error,Training set error,Epoch,Optimization Id')
        self.only_change = only_change
        self.previous_generation = -1

    def write_status(self, id, status):
        if self.only_change == False or self.previous_generation != status.generation:
            msg = ("%s,%s,%s,%s,%f,%f,%d,%s") % (
                datetime.now(),
                status.state,
                status.generation,
                status.total_generations,
                status.validation_set_error,
                status.training_set_error,
                status.epoch,
                id
            )
            self.write_string(msg)
            self.previous_generation = status.generation

    def write_string(self, msg):
        if isinstance(self.log_file, str):
            with open(self.log_file, mode='a', encoding='utf-8', buffering=1) as f:
                f.write(msg+'\n')
        else:
            self.log_file.write(msg+'\n')
            self.log_file.flush()
