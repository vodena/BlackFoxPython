import sys
from contextlib import contextmanager


@contextmanager
def except_handler(exc_handler):
    sys.excepthook = exc_handler
    yield
    sys.excepthook = sys.__excepthook__


def exchandler(type, value, traceback):
    print(': '.join([str(type.__name__), str(value)]))


def validate_ranges(config):
    for r in config.input_ranges:
        if r.min > r.max:
            with except_handler(exchandler):
                raise ValueError(
                    "Maximum value must be greater than minimum value.")
    for r in config.output_ranges:
        if r.min > r.max:
            with except_handler(exchandler):
                raise ValueError(
                    "Maximum value must be greater than minimum value.")


def validate_train_keras(config):
    if len(config.hidden_layer_configs) == 0:
        with except_handler(exchandler):
            raise ValueError(
                "Number of hidden layers must be greater than zero.")
    if not isinstance(config.max_epoch, int):
        with except_handler(exchandler):
            raise ValueError("Number of epochs must be an integer.")
    if not(config.max_epoch > 0):
        with except_handler(exchandler):
            raise ValueError("Number of epochs must be greater than zero.")
    if not (config.validation_split >= 0 and config.validation_split <= 1):
        with except_handler(exchandler):
            raise ValueError("Validation split must be between 0 and 1.")
    for hlc in config.hidden_layer_configs:
        if hlc.neuron_count < 0:
            with except_handler(exchandler):
                raise ValueError(
                    "Number of neurons must be greater than zero.")
    for r in config.input_ranges:
        if r.min > r.max:
            with except_handler(exchandler):
                raise ValueError(
                    "Maximum value must be greater than minimum value.")
    for r in config.output_layer.ranges:
        if r.min > r.max:
            with except_handler(exchandler):
                raise ValueError(
                    "Maximum value must be greater than minimum value.")


def validate_predict_from_file_keras(config):
    return validate_ranges(config)


def validate_test_predict_array_keras(config):
    return validate_ranges(config)


def validate_optimize_keras(config):
    if config.engine_config.proc_timeout_miliseconds < 0:
        with except_handler(exchandler):
            raise ValueError("Values in seconds must be positive.")
    if(config.dropout.min > config.dropout.max):
        with except_handler(exchandler):
            raise ValueError(
                "Maximum value must be greater than minimum value.")
    if(config.hidden_layer_count_range.min >
       config.hidden_layer_count_range.max):
        with except_handler(exchandler):
            raise ValueError(
                "Maximum value must be greater than minimum value.")
    if not(config.hidden_layer_count_range.min > 0):
        with except_handler(exchandler):
            raise ValueError(
                "Number of hidden layers must be greater than zero.")
    if(config.neurons_per_layer.min > config.neurons_per_layer.max):
        with except_handler(exchandler):
            raise ValueError(
                "Maximum value must be greater than minimum value.")
    if not(config.neurons_per_layer.min > 0):
        with except_handler(exchandler):
            raise ValueError("Number of neurons must be greater than zero.")
    if not(config.validation_split >= 0 and config.validation_split <= 1):
        with except_handler(exchandler):
            raise ValueError("Validation split must be between 0 and 1.")
    if not(config.max_epoch > 0):
        with except_handler(exchandler):
            raise ValueError("Number of epochs must be greater than zero.")
    if not (isinstance(config.max_epoch, int)):
        with except_handler(exchandler):
            raise ValueError("Number of epochs must be an integer.")
    validate_ranges(config)


def validate_optimization(func):
    def decorator(*args, **kwargs):
        validate_optimize_keras(kwargs["config"])
        func(*args, **kwargs)
    return decorator


def validate_training(func):
    def decorator(*args, **kwargs):
        validate_train_keras(kwargs["config"])
        func(*args, **kwargs)
    return decorator


def validate_prediction_file(func):
    def decorator(*args, **kwargs):
        validate_predict_from_file_keras(kwargs["config"])
        func(*args, **kwargs)
    return decorator


def validate_prediction_array(func):
    def decorator(*args, **kwargs):
        validate_test_predict_array_keras(kwargs["config"])
        func(*args, **kwargs)
    return decorator
