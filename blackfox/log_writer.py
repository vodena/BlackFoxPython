from datetime import datetime
import sys

class LogWriter(object):
    """LogWriter provides logging capabilities for an ongoing Black Fox optimization.

    Parameters
    ----------
    file : str
        Optional file or sys.stdout used for logging

    """

    def __init__(self, file=sys.stdout):
        self.log_file = file

    def write_neural_network_statues(self, id, statuses):
        status = statuses[-1]
        if len(statuses) >= 2 or (status.validation_set_error > 0 and status.training_set_error > 0):
            msg = ("%s - %s, "
                    "Generation: %s/%s, "
                    "Validation set %s: %f%s, "
                    "Training set %s: %f%s, "
                    "Epoch: %d, "
                    "Optimization Id: %s") % (
                datetime.now(),
                status.state,
                status.generation,
                status.total_generations,
                status.metric_name,
                status.validation_set_error,
                ' %' if status.metric_name == 'MAPE' else '',
                status.metric_name,
                status.training_set_error,
                ' %' if status.metric_name == 'MAPE' else '',
                status.epoch,
                id
            )
            self.write_string(msg)
        else: 
            self.write_string("Evaluating initial models")

    def write_random_forest_statues(self, id, statuses):
        status = statuses[-1]
        if len(statuses) >= 2 or (status.validation_set_error > 0 and status.training_set_error > 0):
            msg = ("%s - %s, "
                "Generation: %s/%s, "
                "Validation set %s: %f, "
                "Training set %s: %f, "
                "Optimization Id: %s") % (
                datetime.now(),
                status.state,
                status.generation,
                status.total_generations,
                status.metric_name,
                status.validation_set_error,
                status.metric_name,
                status.training_set_error,
                id
            )
            self.write_string(msg)
        else: 
            self.write_string("Evaluating initial models")

    def write_xgboost_statues(self, id, statuses):
        self.write_random_forest_statues(id, statuses)

    def write_string(self, msg):
        if isinstance(self.log_file, str):
            with open(self.log_file, mode='a', encoding='utf-8', buffering=1) as f:
                f.write(msg+'\n')
        else:
            self.log_file.write(msg+'\n')
            self.log_file.flush()
