from datetime import datetime
import sys


class CsvLogWriter(object):

    def __init__(self, file=sys.stdout, only_change=True, clear_file=True):
        self.log_file = file
        self.write_string(
            'Time,Status,Generation,Total generations,Validation set error,Training set error,Generation time [s],StartTime,End time,Optimization Id, Epoch', clear_file)
        self.only_change = only_change
        self.previous_generation = -1

    def write_neural_network_statues(self, id, statuses, metric):
        if len(statuses) >= 2:
            status = statuses[-2]
            if self.only_change == False or self.previous_generation != status.generation:
                self.__write_nn_status(id, status)
                if statuses[-1].state != 'Active':
                    self.__write_nn_status(id, statuses[-1])
                self.previous_generation = status.generation

    def __write_nn_status(self, id, status):
        msg = ("%s,%s,%d,%d,%f,%f,%d,%s,%s,%s,%d") % (
            datetime.now(),
            status.state,
            status.generation,
            status.total_generations,
            status.validation_set_error,
            status.training_set_error,
            status.generation_seconds,
            status.start_date_time,
            status.estimated_date_time,
            id,
            status.epoch
        )
        self.write_string(msg)

    def write_random_forest_statues(self, id, statuses, metric):
        if len(statuses) >= 2:
            status = statuses[-2]
            if self.only_change == False or self.previous_generation != status.generation:
                self.__write_rf_status(id, status)
                if statuses[-1].state != 'Active':
                    self.__write_rf_status(id, statuses[-1])
                self.previous_generation = status.generation

    def __write_rf_status(self, id, status):
        msg = ("%s,%s,%d,%d,%f,%f,%d,%s,%s,%s") % (
            datetime.now(),
            status.state,
            status.generation,
            status.total_generations,
            status.validation_set_error,
            status.training_set_error,
            status.generation_seconds,
            status.start_date_time,
            status.estimated_date_time,
            id
        )
        self.write_string(msg)

    def write_xgboost_statues(self, id, statuses, metric):
        self.write_random_forest_statues(id, statuses, metric)

    def write_string(self, msg, clear=False):
        if isinstance(self.log_file, str):
            mode = 'w' if clear else 'a'
            with open(self.log_file, mode=mode, encoding='utf-8', buffering=1) as f:
                f.write(msg+'\n')
        else:
            self.log_file.write(msg+'\n')
            self.log_file.flush()
