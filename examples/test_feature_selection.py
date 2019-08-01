from blackfox import BlackFox
from blackfox import KerasOptimizationConfig, OptimizationEngineConfig
from blackfox import InputConfig, Range
import csv

blackfox_url = 'http://localhost:50476/'
bf = BlackFox(blackfox_url)

input_columns = 9
input_set = []
output_set = []

with open('data/cancer_training_set.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print('Column names are ' + (", ".join(row)))
        else:
            data = list(map(float, row))
            input_set.append(data[:input_columns])
            output_set.append(data[input_columns:])

        line_count += 1

    print('Processed ' + str(line_count) + ' lines.')

config = KerasOptimizationConfig(
    inputs=[
        InputConfig(is_optional=True),
        InputConfig(is_optional=True),
        InputConfig(is_optional=True),
        InputConfig(Range(min=0, max=1)),
        InputConfig(is_optional=True),
        InputConfig(Range(min=0, max=1), is_optional=False),
        InputConfig(is_optional=True),
        InputConfig(Range(min=0, max=1)),
        InputConfig()
    ]
)
# Use CTRL + C to stop optimization
network_stream = bf.optimize_keras_sync(
    input_set=input_set,
    output_set=output_set,
    config=config,
    network_path='data/cancer_feature_selection.h5'
)
