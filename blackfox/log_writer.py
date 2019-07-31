from datetime import datetime
import sys

class LogWriter(object):

    def __init__(self, log_file=sys.stdout):
        self.log_file = log_file

    def write_status(self, id, status):

        msg = ("%s -> %s, "
               "Generation: %s/%s, "
               "Validation set error: %f, "
               "Training set error: %f, "
               "Epoch: %d, "
               "Optimization Id: %s") % (
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

    def write_string(self, msg):
        if isinstance(self.log_file, str):
            with open(self.log_file, mode='a', encoding='utf-8', buffering=1) as f:
                f.write(msg+'\n')
        else:
            self.log_file.write(msg+'\n')
            self.log_file.flush()
