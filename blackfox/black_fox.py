from tempfile import NamedTemporaryFile
from io import BytesIO
import os
import sys
import signal
import time
import shutil
import hashlib
from blackfox import ApiException, Configuration, ApiClient, DataSetApi, NetworkApi, PredictionApi, TrainingApi, OptimizationApi, RecurrentOptimizationApi, KerasOptimizationConfig, KerasRecurrentOptimizationConfig, KerasSeriesOptimizationConfig, InputConfig, Range
from blackfox.log_writer import LogWriter
from blackfox.validation import (validate_training,
                                 validate_optimization,
                                 validate_prediction_file,
                                 validate_prediction_array)


BUF_SIZE = 65536  # lets read stuff in 64kb chunks!


class BlackFox:

    """BlackFox provides methods for neural network parameter optimization.

    Parameters
    ----------
    host : str
        Web API url

    """

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

    def __log_status(self, log_writer, id, status, metric):
        if log_writer is not None:
            log_writer.write_status(id, status, metric)

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
        """Starts the optimization.

        Performs the Black Fox optimization and finds the best parameters and hyperparameters of a target model neural network.

        Parameters
        ----------
        input_set : str
            Input data (x train data)
        output_set : str
            Output data (y train data or target data)
        integrate_scaler : bool
            If True, Black Fox will integrate a scaler function used for data scaling/normalization in the model
        network_type : str
            Optimized model file format (.h5 | .onnx | .pb)
        data_set_path : str
            Optional .csv file used instead of input_set/output_set as a source for training data
        config : KerasSeriesOptimizationConfig
            Configuration for Black Fox optimization
        network_path : str
            Save path for the optimized NN; will be used after the function finishes to automatically save optimized network
        status_interval : int
            Time interval for repeated server calls for optimization info and logging
        log_writer : str
            Optional log writer used for logging the optimization process

        Returns
        -------
        (BytesIO, KerasOptimizedNetwork, dict)
            byte array from network model, optimized network info, network metadata
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
        """Starts the optimization.

        Performs the Black Fox optimization for timeseries data and finds the best parameters and hyperparameters of a target model neural network.

        Parameters
        ----------
        input_set : str
            Input data (x train data)
        output_set : str
            Output data (y train data or target data)
        integrate_scaler : bool
            If True, Black Fox will integrate a scaler function used for data scaling/normalization in the model
        network_type : str
            Optimized model file format (.h5 | .onnx | .pb)
        data_set_path : str
            Optional .csv file used instead of input_set/output_set as a source for training data
        config : KerasSeriesOptimizationConfig
            Configuration for Black Fox optimization
        network_path : str
            Save path for the optimized NN; will be used after the function finishes to automatically save optimized network
        status_interval : int
            Time interval for repeated server calls for optimization info and logging
        log_writer : str
            Optional log writer used for logging the optimization process

        Returns
        -------
        (BytesIO, KerasOptimizedNetwork, dict)
            byte array from network model, optimized network info, network metadata
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

        if is_series:
            if len(config.inputs) != len(config.input_window_range_configs):
                raise Exception('Number of input columns is not same as input_window_range_configs')
            if len(config.output_ranges) != len(config.output_window_configs):
                raise Exception('Number of output columns is not same as output_window_configs')

        if config.hidden_layer_count_range is None:
            config.hidden_layer_count_range = Range(1, 15)

        if config.neurons_per_layer is None:
            if config.inputs is None or config.output_ranges is None:
                config.neurons_per_layer = Range(1, 10)
            else:
                avg_count = int(len(config.inputs) + len(config.output_ranges)) / 2
                min_neurons = int(avg_count / 3)
                max_neurons = int(avg_count * 3)
                if min_neurons <= 0:
                    min_neurons = 1
                if max_neurons < 10:
                    max_neurons = 10
                config.neurons_per_layer = Range(min_neurons, max_neurons)

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
                self.__log_string(
                    log_writer, "Uploading data set " + data_set_path)
            config.dataset_id = self.upload_data_set(data_set_path)

        if tmp_file is not None:
            os.remove(tmp_file.name)

        self.__log_string(log_writer, "Starting...")
        if is_series:
            id = self.optimization_api.post_series(config=config)
        else:
            id = self.optimization_api.post(config=config)

        def signal_handler(sig, frame):
            self.__log_string(log_writer, "Stopping optimization : "+id)
            if hasattr(log_writer, 'log_file') is False or log_writer.log_file is not sys.stdout:
                print("Stopping optimization : "+id)
            self.stop_optimization_keras(id)

        signal.signal(signal.SIGINT, signal_handler)

        running = True
        status = None
        while running:
            try:
                status = self.optimization_api.get_status(id)
                running = (status.state == 'Active')
                metric = 'error' if config.problem_type != 'BinaryClassification' else config.binary_optimization_metric
                self.__log_status(log_writer, id, status, metric)
            except Exception as e:
                self.__log_string(log_writer, "Error: " + str(e))
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
                metadata = self.network_api.get_metadata(status.network.id)
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
        config=KerasOptimizationConfig(),
        data_set_path=None
    ):
        """Async optimization call.

        Performs the Black Fox optimization asynchronously (non-blocking), so the user must querry the server periodically in order to get the progress info.

        Parameters
        ----------
        config : KerasOptimizationConfig
            Configuration for Black Fox optimization
        data_set_path : str
            Path to a .csv file holding the training dataset

        Returns
        -------
        str
            Optimization process id
        """
        # validate_optimize_keras(config)
        if data_set_path is not None:
            config.dataset_id = self.upload_data_set(data_set_path)
        return self.optimization_api.post(config=config)

    def get_optimization_status_keras(
        self,
        id,
        integrate_scaler=False,
        network_type='h5',
        network_path=None
    ):
        """Gets current async optimization status.

        Query of the current optimization status when it is performed asynchronously.

        Parameters
        ----------
        id : str
            Optimization process id (i.e. from optimize_keras method)
        integrate_scaler : bool
            If True, Black Fox will integrate a scaler function used for data scaling/normalization in the model
        network_type : str
            Optimized model file format (.h5 | .onnx | .pb)
        network_path : str
            Save path for the optimized NN; will be used after the function finishes to automatically save optimized network

        Returns
        -------
        KerasOptimizationStatus
            An object depicting the current optimization status
        """
        status = self.optimization_api.get_status(id)
        if (
            (status.state == 'Finished' or status.state == 'Stopped')
            and (network_path is not None and status.network is not None and status.network.id is not None)
        ):
            self.download_network(
                status.network.id,
                integrate_scaler=integrate_scaler,
                network_type=network_type,
                path=network_path)

        return status

    def cancel_optimization_keras(self, id):
        self.optimization_api.post_action(id, 'Cancel')

    def stop_optimization_keras(self, id, network_path=None):
        """Stops current async optimization.

        Sends a request for stopping the ongoing optimization, and returns the current best solution.

        Parameters
        ----------
        id : str
            Optimization process id (i.e. from optimize_keras method)
        network_path : str
            Save path for the optimized NN; will be used after the function finishes to automatically save optimized network

        Returns
        -------
        KerasOptimizationStatus
            An object depicting the current optimization status
        """
        self.optimization_api.post_action(id, 'Stop')
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
        """Starts the optimization.

        Performs the Black Fox optimization using recurrent neural networks and finds the best parameters and hyperparameters of a target model.

        Parameters
        ----------
        input_set : str
            Input data (x train data)
        output_set : str
            Output data (y train data or target data)
        integrate_scaler : bool
            If True, Black Fox will integrate a scaler function used for data scaling/normalization in the model
        network_type : str
            Optimized model file format (.h5 | .onnx | .pb)
        data_set_path : str
            Optional .csv file used instead of input_set/output_set as a source for training data
        config : KerasRecurrentOptimizationConfig
            Configuration for Black Fox optimization
        network_path : str
            Save path for the optimized NN; will be used after the function finishes to automatically save optimized network
        status_interval : int
            Time interval for repeated server calls for optimization info and logging
        log_writer : str
            Optional log writer used for logging the optimization process

        Returns
        -------
        (BytesIO, KerasOptimizedNetwork, dict)
            byte array from network model, optimized network info, network metadata
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

        if config.hidden_layer_count_range is None:
            config.hidden_layer_count_range = Range(1, 15)

        if config.neurons_per_layer is None:
            if config.inputs is None or config.output_ranges is None:
                config.neurons_per_layer = Range(1, 10)
            else:
                avg_count = int(len(config.inputs) + len(config.output_ranges)) / 2
                min_neurons = int(avg_count / 3)
                max_neurons = int(avg_count * 3)
                if min_neurons <= 0:
                    min_neurons = 1
                if max_neurons < 10:
                    max_neurons = 10
                config.neurons_per_layer = Range(min_neurons, max_neurons)

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
            self.__log_status(log_writer, id, status, 'error')
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
                metadata = self.network_api.get_metadata(status.network.id)
                return byte_io, status.network, metadata
            else:
                return None, None, None

        elif status.state == 'Error':
            self.__log_string(log_writer, "Optimization error")
        else:
            self.__log_string(log_writer, "Unknown error")

        return None, None, None

    def get_metadata(self, network_path):
        """Network metadata retrieval

        Gets the neural network metadata from a network file.

        Parameters
        ----------
        network_path : str
            Load path for the neural network file from wich the metadata would be read

        Returns
        -------
        dict
            network metadata
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

        return self.network_api.get_metadata(id)

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
