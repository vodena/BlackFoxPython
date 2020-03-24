from datetime import datetime
import sys

class LogWriter(object):
    """LogWriter provides logging capabilities for an ongoing Black Fox optimization.

    Parameters
    ----------
    log_file : str
        Optional file or sys.stdout used for logging

    """

    def __init__(self, log_file=sys.stdout):
        self.log_file = log_file

    def write_status(self, id, statuses, metric):
        status = statuses[-1]
        msg = ("%s -> %s, "
               "Generation: %s/%s, "
               "Validation set %s: %f, "
               "Training set %s: %f, "
               "Epoch: %d, "
               "Optimization Id: %s") % (
            datetime.now(),
            status.state,
            status.generation,
            status.total_generations,
            metric,
            status.validation_set_error,
            metric,
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
