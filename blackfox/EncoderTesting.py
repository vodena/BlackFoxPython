# from blackfox import InputConfig, Range
from blackfox_restapi.models.input_config import InputConfig
from blackfox_restapi.models.range import Range

def __fill_inputs(inputs, data_set):

    if inputs is None or len(inputs) == 0:
        inputs = []
        for row in data_set:
            for i, d in enumerate(row):
                if len(inputs) <= i or inputs[i] is None:
                    inputs.append(InputConfig(range = Range(d, d), encoding = None))
                else:
                    r = inputs[i].range
                    r.min = min(r.min, d)
                    r.max = max(r.max, d)
    else:
        if len(inputs) < len(data_set[0]) or len(inputs) > len(data_set[0]):
            raise Exception ("The number of encoding types must match the number of input variables")
        for j in range(len(inputs)):
            inp = inputs[j]
            if inp.range is None:
                inp.range = Range()
                inp.range.min = data_set[0][j]
                inp.range.max = data_set[0][j]
                for row in data_set:
                    d = row[j]
                    inp.range.min = min(inp.range.min, d)
                    inp.range.max = max(inp.range.max, d)
    
    for input in inputs:
        if input.encoding is None:
            if isinstance(input.range.max, str) or isinstance(input.range.min, str):
                input.encoding = 'Target'
                input.range.max = None
                input.range.min = None
            else:
                input.encoding = 'None'
        elif input.encoding is not 'None' and (isinstance(input.range.max, str) or isinstance(input.range.min, str)):
            input.range.max = None
            input.range.min = None
    return inputs

data = [['prvi', 1], ['prvi', 1], ['drugi', 2], ['treci', 3]]
inputs = [InputConfig(encoding='Target'), InputConfig(encoding='Target')]

__fill_inputs(inputs, data)


