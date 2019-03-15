from contracts import contract, new_contract

def validate_ranges(config):
    for r in config.input_ranges:
        if r.min > r.max:
            return False
    for r in config.output_ranges:
        if r.min > r.max:
            return False
    return True

def validate_train_keras(config):
    if len(config.hidden_layer_configs) == 0:
        return False
    if not isinstance(config.max_epoch, int) or not(config.max_epoch > 0):
        return False
    if not (config.validation_split >= 0 and config.validation_split <= 1):
        return False
    for hlc in config.hidden_layer_configs:
        if hlc.neuron_count<0:
            return False
    for r in config.input_ranges:
        if r.min > r.max:
            return False
    for r in config.output_layer.ranges:
        if r.min > r.max:
            return False
    return True

def validate_predict_from_file_keras(config):
    return validate_ranges(config)

def validate_test_predict_array_keras(config):
    return validate_ranges(config)

def validate_optimize_keras(config):
    if config.engine_config.proc_timeout_miliseconds<0:
        return False
    if(config.dropout.min > config.dropout.max):
        return False
    if(config.hidden_layer_count_range.min > config.hidden_layer_count_range.max):
        return False
    if not(config.hidden_layer_count_range.min>0):
        return False
    if(config.neurons_per_layer.min > config.neurons_per_layer.max):
        return False
    if not(config.neurons_per_layer.min>0):
        return False
    if not(config.validation_split>0 and config.validation_split<1):
        return False
    if not(config.max_epoch>0)  or not (isinstance(config.max_epoch, int)):
        return False
    return validate_ranges(config)

new_contract('train_keras_validation', lambda config: validate_train_keras(config)  )
new_contract('predict_from_file_keras_validation', lambda config: validate_predict_from_file_keras(config)  )
new_contract('predict_array_keras_validation', lambda config: validate_test_predict_array_keras(config)  )
new_contract('optimize_keras_validation', lambda config: validate_optimize_keras(config)  )


