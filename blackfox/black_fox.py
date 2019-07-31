from tempfile import NamedTemporaryFile
from io import BytesIO
import os
import sys
import signal
import time
import shutil
import hashlib
from blackfox.rest import ApiException
from blackfox.configuration import Configuration
from blackfox.api_client import ApiClient
from blackfox.api.data_set_api import DataSetApi
from blackfox.api.network_api import NetworkApi
from blackfox.api.prediction_api import PredictionApi
from blackfox.api.training_api import TrainingApi
from blackfox.api.optimization_api import OptimizationApi
from blackfox.api.recurrent_optimization_api import RecurrentOptimizationApi
from blackfox.models.keras_optimization_config import KerasOptimizationConfig
from blackfox.log_writer import LogWriter
from blackfox.models.keras_recurrent_optimization_config import KerasRecurrentOptimizationConfig
from blackfox.models.keras_series_optimization_config import KerasSeriesOptimizationConfig
from blackfox.models.input_config import InputConfig
from blackfox.models.range import Range
from blackfox.validation import (validate_training,
                                 validate_optimization,
                                 validate_prediction_file,
                                 validate_prediction_array)


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
        self.recurrent_optimization_api = RecurrentOptimizationApi(self.client)

    def __log_string(self, log_writer, msg):
        if log_writer is not None:
            log_writer.write_string(msg)

    def __log_status(self, log_writer, id, status):
        if log_writer is not None:
            log_writer.write_status(id, status)

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

    def download_network(
        self, id, integrate_scaler=False,
        network_type='h5', path=None
    ):
        temp_path = self.network_api.get(
            id, integrate_scaler=integrate_scaler, network_type=network_type)
        if path is None:
            return open(temp_path, 'rb')
        else:
            shutil.move(temp_path, path)

    @validate_training
    def train_keras(
        self,
        config,
        integrate_scaler=False,
        network_type='h5',
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
            self.download_network(
                trained_network.id,
                integrate_scaler=integrate_scaler,
                network_type=network_type,
                path=network_path)

        return trained_network

    @validate_prediction_file
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

    @validate_prediction_array
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

    def __get_ranges(self, data_set):
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

    def __fill_inputs(self, inputs, data_set):
        if inputs is None or len(inputs) == 0:
            inputs = []
            for row in data_set:
                for i, d in enumerate(row):
                    if len(inputs) <= i or inputs[i] is None:
                        inputs.append(InputConfig(range=Range(d, d)))
                    else:
                        r = inputs[i].range
                        r.min = min(r.min, d)
                        r.max = max(r.max, d)
        else:
            for j in range(len(inputs)):
                inp = inputs[j]
                if inp.range is None:
                    inp.range = Range(float("inf"), float("-inf"))
                    for row in data_set:
                        d = row[j]
                        inp.range.min = min(inp.range.min, d)
                        inp.range.max = max(inp.range.max, d)
        return inputs

    def optimize_keras_sync(
        self,
        input_set=None,
        output_set=None,
        integrate_scaler=False,
        network_type='h5',
        data_set_path=None,
        config=KerasOptimizationConfig(),
        network_path=None,
        status_interval=5,
        log_writer=LogWriter()
    ):
        """
        Find optimal network for given problem.
        :param KerasOptimizationConfig config:
        :param str input_set:
        :param str output_set:
        :param bool integrate_scaler:
        :param str network_type:
        :param str data_set_path:
        :param str network_path:
        :param int status_interval:
        :param str log_writer:
        :return: (BytesIO, KerasOptimizedNetwork): byte array from network model, optimized network info
                If data_set_path is not None upload data set,
                and sets config.dataset_id to new id.
                If network_path is not None download network to given file.
                If log_writer is not None write to log file 
                every 5 seconds(status_interval)
        """
        return self.__optimize_keras_sync(
            is_series=False,
            input_set=input_set,
            output_set=output_set,
            integrate_scaler=integrate_scaler,
            network_type=network_type,
            data_set_path=data_set_path,
            config=config,
            network_path=network_path,
            status_interval=status_interval,
            log_writer=log_writer
        )

    def optimize_series_keras_sync(
        self,
        input_set=None,
        output_set=None,
        integrate_scaler=False,
        network_type='h5',
        data_set_path=None,
        config=KerasSeriesOptimizationConfig(),
        network_path=None,
        status_interval=5,
        log_writer=LogWriter()
    ):
        """
        Find optimal network for given problem.
        :param KerasSeriesOptimizationConfig config:
        :param str input_set:
        :param str output_set:
        :param bool integrate_scaler:
        :param str network_type:
        :param str data_set_path:
        :param str network_path:
        :param int status_interval:
        :param str log_writer:
        :return: (BytesIO, KerasOptimizedNetwork, dict): byte array from network model, optimized network info, metadata
                If data_set_path is not None upload data set,
                and sets config.dataset_id to new id.
                If network_path is not None download network to given file.
                If log_writer is not None write to log file 
                every 5 seconds(status_interval)
        """
        return self.__optimize_keras_sync(
            is_series=True,
            input_set=input_set,
            output_set=output_set,
            integrate_scaler=integrate_scaler,
            network_type=network_type,
            data_set_path=data_set_path,
            config=config,
            network_path=network_path,
            status_interval=status_interval,
            log_writer=log_writer
        )

    def __optimize_keras_sync(
        self,
        is_series=False,
        input_set=None,
        output_set=None,
        integrate_scaler=False,
        network_type='h5',
        data_set_path=None,
        config=None,
        network_path=None,
        status_interval=5,
        log_writer=LogWriter()
    ):
        print('Use CTRL + C to stop optimization')
        tmp_file = None
        if input_set is not None and output_set is not None:
            if type(input_set) is not list:
                input_set = input_set.tolist()
            if type(output_set) is not list:
                output_set = output_set.tolist()
            tmp_file = NamedTemporaryFile(delete=False)
            # input ranges
            config.inputs = self.__fill_inputs(config.inputs, input_set)
            # output ranges
            if config.output_ranges is None:
                config.output_ranges = self.__get_ranges(output_set)
            data_set = list(map(lambda x, y: (','.join(map(str, x)))+',' +
                                (','.join(map(str, y))), input_set, output_set))

            column_count = len(config.inputs) + len(config.output_ranges)
            column_range = range(0, column_count)
            headers = map(lambda i: 'column_'+str(i), column_range)
            data_set.insert(0, ','.join(headers))
            csv = '\n'.join(data_set)
            tmp_file.write(csv.encode("utf-8"))
            tmp_file.close()
            if data_set_path is not None:
                self.__log_string(log_writer, 'Ignoring data_set_path')
            data_set_path = str(tmp_file.name)

        if data_set_path is not None:
            if config.inputs is None:
                self.__log_string(log_writer, "config.inputs is None")
                return None, None, None
            if config.output_ranges is None:
                self.__log_string(log_writer, "config.output_ranges is None")
                return None, None, None
            if tmp_file is not None:
                self.__log_string(log_writer, "Uploading data set")
            else:
                self.__log_string(log_writer, "Uploading data set " + data_set_path)
            config.dataset_id = self.upload_data_set(data_set_path)

        if tmp_file is not None:
            os.remove(tmp_file.name)

        self.__log_string(log_writer, "Starting...")
        if is_series:
            id = self.optimization_api.post_series_async(config=config)
        else:
            id = self.optimization_api.post_async(config=config)

        def signal_handler(sig, frame):
            self.__log_string(log_writer, "Stopping optimization : "+id)
            if hasattr(log_writer, 'log_file') is False or log_writer.log_file is not sys.stdout:
                print("Stopping optimization : "+id)
            self.stop_optimization_keras(id)

        signal.signal(signal.SIGINT, signal_handler)

        running = True
        status = None
        while running:
            status = self.optimization_api.get_status_async(id)
            running = (status.state == 'Active')
            self.__log_status(log_writer, id, status)
            time.sleep(status_interval)

        if status.state == 'Finished' or status.state == 'Stopped':
            print('Optimization state: ', status.state)
            if status.network is not None and status.network.id is not None:
                self.__log_string(log_writer,
                         "Downloading network " +
                         status.network.id)
                network_stream = self.download_network(
                    status.network.id,
                    integrate_scaler=integrate_scaler,
                    network_type=network_type
                )
                data = network_stream.read()
                if network_path is not None:
                    self.__log_string(log_writer,
                             "Saving network " +
                             status.network.id + " to " + network_path)
                    with open(network_path, 'wb') as f:
                        f.write(data)
                byte_io = BytesIO(data)
                metadata = self.network_api.metadata(status.network.id)
                return byte_io, status.network, metadata
            else:
                return None, None, None

        elif status.state == 'Error':
            self.__log_string(log_writer, "Optimization error")
        else:
            self.__log_string(log_writer, "Unknown error")

        return None, None, None

    @validate_optimization
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
        # validate_optimize_keras(config)
        if data_set_path is not None:
            config.dataset_id = self.upload_data_set(data_set_path)
        return self.optimization_api.post_async(config=config)

    def get_optimization_status_keras(
        self,
        id,
        integrate_scaler=False,
        network_type='h5',
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
            self.download_network(
                status.network.id,
                integrate_scaler=integrate_scaler,
                network_type=network_type,
                path=network_path)

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

    def optimize_recurrent_keras_sync(
        self,
        input_set=None,
        output_set=None,
        integrate_scaler=False,
        network_type='h5',
        data_set_path=None,
        config=None,
        network_path=None,
        status_interval=5,
        log_writer=LogWriter()
    ):
        """
        Find optimal network for given problem.
        :param KerasRecurrentOptimizationConfig config:
        :param str input_set:
        :param str output_set:
        :param bool integrate_scaler:
        :param str network_type:
        :param str data_set_path:
        :param str network_path:
        :param int status_interval:
        :param str log_writer:
        :return: (BytesIO, KerasRecurrentOptimizedNetwork, dict): byte array from network model, optimized network info, metadata
                If data_set_path is not None upload data set,
                and sets config.dataset_id to new id.
                If network_path is not None download network to given file.
                If log_writer is not None write to log file 
                every 5 seconds(status_interval)
        """
        print('Use CTRL + C to stop optimization')
        tmp_file = None
        if input_set is not None and output_set is not None:
            if type(input_set) is not list:
                input_set = input_set.tolist()
            if type(output_set) is not list:
                output_set = output_set.tolist()
            tmp_file = NamedTemporaryFile(delete=False)
            # input ranges
            config.inputs = self.__fill_inputs(config.inputs, input_set)
            # output ranges
            if config.output_ranges is None:
                config.output_ranges = self.__get_ranges(output_set)
            data_set = list(map(lambda x, y: (','.join(map(str, x)))+',' +
                                (','.join(map(str, y))), input_set, output_set))

            column_count = len(config.inputs) + len(config.output_ranges)
            column_range = range(0, column_count)
            headers = map(lambda i: 'column_'+str(i), column_range)
            data_set.insert(0, ','.join(headers))
            csv = '\n'.join(data_set)
            tmp_file.write(csv.encode("utf-8"))
            tmp_file.close()
            if data_set_path is not None:
                self.__log_string(log_writer, 'Ignoring data_set_path')
            data_set_path = str(tmp_file.name)

        if data_set_path is not None:
            if config.inputs is None:
                self.__log_string(log_writer, "config.inputs is None")
                return None, None, None
            if config.output_ranges is None:
                self.__log_string(log_writer, "config.output_ranges is None")
                return None, None, None
            if tmp_file is not None:
                self.__log_string(log_writer, "Uploading data set")
            else:
                self.__log_string(log_writer, "Uploading data set " +
                         data_set_path)
            config.dataset_id = self.upload_data_set(data_set_path)

        if tmp_file is not None:
            os.remove(tmp_file.name)

        self.__log_string(log_writer, "Starting...")
        id = self.recurrent_optimization_api.post(config=config)

        def signal_handler(sig, frame):
            self.__log_string(log_writer, "Stopping optimization : "+id)
            if hasattr(log_writer, 'log_file') is False or log_writer.log_file is not sys.stdout:
                print("Stopping optimization : "+id)
            self.stop_optimization_keras(id)

        signal.signal(signal.SIGINT, signal_handler)

        running = True
        status = None
        while running:
            status = self.recurrent_optimization_api.get_status(id)
            running = (status.state == 'Active')
            if log_writer is not None:
                self.__log_string(
                    log_writer,
                    ("%s -> %s, "
                     "Generation: %s/%s, "
                     "Validation set error: %f, "
                     "Training set error: %f, "
                     "Epoch: %d, "
                     "Optimization Id: %s") %
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
            print('Optimization state: ', status.state)
            if status.network is not None and status.network.id is not None:
                self.__log_string(log_writer,
                         "Downloading network " +
                         status.network.id)
                network_stream = self.download_network(
                    status.network.id,
                    integrate_scaler=integrate_scaler,
                    network_type=network_type
                )
                data = network_stream.read()
                if network_path is not None:
                    self.__log_string(log_writer,
                             "Saving network " +
                             status.network.id + " to " + network_path)
                    with open(network_path, 'wb') as f:
                        f.write(data)
                byte_io = BytesIO(data)
                metadata = self.network_api.metadata(status.network.id)
                return byte_io, status.network, metadata
            else:
                return None, None, None

        elif status.state == 'Error':
            self.__log_string(log_writer, "Optimization error")
        else:
            self.__log_string(log_writer, "Unknown error")

        return None, None, None

    def get_metadata(self, network_path):
        """
        Get network metadata.

        :param str or BytesIO network_path:
        :return: dict: 
        """
        id = None
        if isinstance(network_path, BytesIO):
            with NamedTemporaryFile(delete=False) as out:
                out.write(network_path.read())
                file_path = str(out.name)
            id = self.upload_network(file_path)
            os.remove(file_path)
        else:
            id = self.upload_network(network_path)

        return self.network_api.metadata(id)

    def convert_to(self, network_path, network_type, network_dst_path=None, integrate_scaler=False):
        id = None
        if isinstance(network_path, BytesIO):
            with NamedTemporaryFile(delete=False) as out:
                out.write(network_path.read())
                file_path = str(out.name)
            id = self.upload_network(file_path)
            os.remove(file_path)
        else:
            id = self.upload_network(network_path)
        stream = self.download_network(
            id,
            network_type=network_type,
            integrate_scaler=integrate_scaler,
            path=network_dst_path
        )
        if stream is not None:
            data = stream.read()
            byte_io = BytesIO(data)
            return byte_io

    def convert_to_onnx(
        self, network_path,
        network_dst_path=None, integrate_scaler=False
    ):
        self.convert_to(network_path, 'onnx', network_dst_path,
                        integrate_scaler=integrate_scaler)

    def convert_to_pb(
        self, network_path,
        network_dst_path=None, integrate_scaler=False
    ):
        self.convert_to(network_path, 'pb', network_dst_path,
                        integrate_scaler=integrate_scaler)

    def sha1(self, path):
        sha1 = hashlib.sha1()
        try:
            with open(path, 'rb') as f:
                while True:
                    data = f.read(BUF_SIZE)
                    if not data:
                        break
                    sha1.update(data)

        except IOError:
            print("File " + path + " doesn't exist.")

        return sha1.hexdigest()
