from blackfox.api.data_set_api import DataSetApi
from blackfox.api.network_api import NetworkApi
from blackfox.api.prediction_api import PredictionApi
from blackfox.api.training_api import TrainingApi
from blackfox.api.optimization_api import OptimizationApi
from blackfox.models.keras_optimization_config import KerasOptimizationConfig
from blackfox.models.range import Range

from blackfox.api_client import ApiClient
from blackfox.configuration import Configuration
from blackfox.rest import ApiException

import hashlib
import shutil
import time
from datetime import datetime
import signal
import sys
import os
from io import BytesIO
from tempfile import NamedTemporaryFile

BUF_SIZE = 65536  # lets read stuff in 64kb chunks!


class BlackFox:

    def __init__(self, host="http://localhost:50476/"):
        self.host = host
        configuration = Configuration()
        configuration.host = host
        self.client = ApiClient(configuration)
        self.data_set_api = DataSetApi(self.client)
        self.network_api = NetworkApi(self.client)
        self.prediction_api = PredictionApi(self.client)
        self.training_api = TrainingApi(self.client)
        self.optimization_api = OptimizationApi(self.client)

    def log(self, stream, msg):
        if isinstance(stream, str):
            with open(stream, mode='a', encoding='utf-8', buffering=1) as f:
                f.write(msg)
        else:
            stream.write(msg)
            stream.flush()

    def upload_data_set(self, path):
        id = self.sha1(path)
        try:
            self.data_set_api.head(id)
        except ApiException as e:
            if e.status == 404:
                id = self.data_set_api.post(file=path)
            else:
                raise e
        return id

    def download_data_set(self, id, path):
        temp_path = self.data_set_api.get(id)
        shutil.move(temp_path, path)

    def upload_network(self, path):
        id = self.sha1(path)
        try:
            self.network_api.head(id)
        except ApiException as e:
            if e.status == 404:
                id = self.network_api.post(file=path)
            else:
                raise e
        return id

    def download_network(self, id, path=None):
        temp_path = self.network_api.get(id)
        if path is None:
            return open(temp_path, 'rb')
        else:
            shutil.move(temp_path, path)

    def train_keras(
        self,
        config,
        data_set_path=None,
        network_path=None
    ):
        """
        Train network

        :param KerasTrainingConfig config:
        :param str data_set_path:
        :param str nework_path:
        :return: TrainedNetwork
                If data_set_path is not None upload data set 
                and sets config.dataset_id to new id.
                If network_path is not None 
                download network to given file.
        """
        if data_set_path is not None:
            config.dataset_id = self.upload_data_set(data_set_path)

        trained_network = self.training_api.post(value=config)

        if network_path is not None:
            self.download_network(trained_network.id, network_path)

        return trained_network

    def predict_from_file_keras(
        self,
        config,
        network_path=None,
        data_set_path=None,
        result_path=None
    ):
        """
        Predict values and download results in file

        :param PredictionFileConfig config:
        :param str network_path:
        :param str data_set_path:
        :param str result_path:
        :return: str: result data set id
                If network_path is not None upload network,
                and sets config.network_id to new id.
                If data_set_path is not None upload data set,
                and sets config.data_set_id to new id.
                If result_path is not None download results
                to given file.
        """
        if network_path is not None:
            config.network_id = self.upload_network(network_path)
        if data_set_path is not None:
            config.data_set_id = self.upload_data_set(data_set_path)
        result_id = self.prediction_api.post_file(config=config)
        if result_path is not None:
            self.download_data_set(result_id, result_path)
        return result_id

    def predict_from_array_keras(
        self,
        config,
        network_path=None
    ):
        """
        Predict values and return results

        :param PredictionArrayConfig config:
        :param str network_path:
        :return: list[list[float]]: 
                If network_path is not None upload network,
                and sets config.network_id to new id.
        """
        if network_path is not None:
            config.network_id = self.upload_network(network_path)
        results = self.prediction_api.post_array(config=config)
        return results

    def get_ranges(self, data_set):
        ranges = []
        for row in data_set:
            for i, d in enumerate(row):
                if len(ranges) <= i or ranges[i] is None:
                    ranges.append(Range(d, d))
                else:
                    r = ranges[i]
                    r.min = min(r.min, d)
                    r.max = max(r.max, d)
        return ranges

    def optimize_keras_sync(
        self,
        input_set=None,
        output_set=None,
        data_set_path=None,
        config=KerasOptimizationConfig(),
        network_path=None,
        status_interval=5,
        log_file=sys.stdout
    ):
        """
        Find optimal network for given problem.

        :param KerasOptimizationConfig config:
        :param str input_set:
        :param str output_set:
        :param str data_set_path:
        :param str network_path:
        :param int status_interval:
        :param str log_file:
        :return: BytesIO: byte array from network model
                If data_set_path is not None upload data set,
                and sets config.dataset_id to new id.
                If network_path is not None download network to given file.
                If log_file is not None write to log file 
                every 5 seconds(status_interval)
        """
        print('Use CTRL + C to stop optimization')

        if input_set is not None and output_set is not None:
            tmp_file = NamedTemporaryFile(delete=False)
            # input ranges
            config.input_ranges = self.get_ranges(input_set)
            # output ranges
            config.output_ranges = self.get_ranges(output_set)
            data_set = list(map(lambda x, y: (','.join(map(str, x)))+',' +
                                (','.join(map(str, y))), input_set, output_set))

            column_count = len(config.input_ranges) + len(config.output_ranges)
            column_range = range(0, column_count)
            headers = map(lambda i: 'column_'+str(i), column_range)
            data_set.insert(0, ','.join(headers))
            csv = '\n'.join(data_set)
            tmp_file.write(csv.encode("utf-8"))
            tmp_file.close()
            if data_set_path is not None:
                self.log(log_file, 'Ignoring data_set_path\n')
            data_set_path = str(tmp_file.name)

        if data_set_path is not None:
            if config.input_ranges is None:
                self.log(log_file, "config.input_ranges is None\n")
                return None
            if config.output_ranges is None:
                self.log(log_file, "config.output_ranges is None\n")
                return None
            if tmp_file is not None:
                self.log(log_file, "Uploading data set\n")
            else:
                self.log(log_file, "Uploading data set " + data_set_path + "\n")
            config.dataset_id = self.upload_data_set(data_set_path)

        if tmp_file is not None:
            os.remove(tmp_file.name)

        self.log(log_file, "Starting...\n")
        id = self.optimization_api.post_async(config=config)

        def signal_handler(sig, frame):
            self.log(log_file, "Stopping optimization : "+id+"\n")
            print("Stopping optimization : "+id)
            self.stop_optimization_keras(id)

        signal.signal(signal.SIGINT, signal_handler)

        running = True
        status = None
        while running:
            status = self.optimization_api.get_status_async(id)
            running = (status.state == 'Active')
            if log_file is not None:
                self.log(
                    log_file,
                    ("%s -> %s, "
                     "Generation: %s/%s, "
                     "Validation set error: %f, "
                     "Training set error: %f, "
                     "Epoch: %d, "
                     "Optimization Id: %s\n") %
                    (
                        datetime.now(),
                        status.state,
                        status.generation,
                        status.total_generations,
                        status.validation_set_error,
                        status.training_set_error,
                        status.epoch,
                        id
                    )
                )
            time.sleep(status_interval)

        if status.state == 'Finished' or status.state == 'Stopped':
            if status.network is not None and status.network.id is not None:
                self.log(log_file,
                         "Downloading network " +
                         status.network.id + "\n")
                network_stream = self.download_network(status.network.id)
                data = network_stream.read()
                if network_path is not None:
                    self.log(log_file,
                             "Saving network " +
                             status.network.id + " to " + network_path + "\n")
                    with open(network_path, 'wb') as f:
                        f.write(data)
                return BytesIO(data)

        elif status.state == 'Error':
            self.log(log_file, "Optimization error\n")
            return None
        else:
            self.log(log_file, "Unknown error\n")
            return None

    def optimize_keras(
        self,
        config,
        data_set_path=None
    ):
        """
        Find optimal network for given problem async.

        :param KerasOptimizationConfig config:
        :param str data_set_path:
        :return: str: 
                If data_set_path is not None upload data set,
                and sets config.dataset_id to new id.
                Return optimization id.
        """
        if data_set_path is not None:
            config.dataset_id = self.upload_data_set(data_set_path)
        return self.optimization_api.post_async(config=config)

    def get_optimization_status_keras(
        self,
        id,
        network_path=None
    ):
        """
        Get status for optimization.

        :param KerasOptimizationConfig config:
        :param str network_path:
        :return: KerasOptimizationStatus: 
                If data_set_path is not None upload data set,
                and sets config.dataset_id to new id.
        """
        status = self.optimization_api.get_status_async(id)
        if (
            (status.state == 'Finished' or status.state == 'Stopped')
            and (network_path is not None)
        ):
            self.download_network(status.network.id, network_path)

        return status

    def cancel_optimization_keras(self, id):
        """
            Cancel optimization.

            :param str id:
            :return: None: 
                    Call get_optimization_status_keras to get status.
        """
        self.optimization_api.post_action_async(id, 'Cancel')

    def stop_optimization_keras(self, id, network_path=None):
        """
            Stop optimization

            :param str id:
            :param str network_path:
            :return: None: 
                    If network_path is not None download network to given file,
                    else call get_optimization_status_keras to get status
                    and download network.
        """
        self.optimization_api.post_action_async(id, 'Stop')
        if network_path is not None:
            state = 'Active'
            while state == 'Active':
                status = self.get_optimization_status_keras(id, network_path)
                state = status.state

    def sha1(self, path):
        sha1 = hashlib.sha1()
        with open(path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha1.update(data)
        return sha1.hexdigest()
