from blackfox import BlackFox
from blackfox import KerasOptimizationConfig
from blackfox import OptimizationEngineConfig
from blackfox import Range
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
            print(f'Column names are {", ".join(row)}')
        else:
            data = list(map(float, row))
            input_set.append(data[:input_columns])
            output_set.append(data[input_columns:])

        line_count += 1

    print(f'Processed {line_count} lines.')

# Use CTRL + C to stop optimization
network_io = bf.optimize_keras_sync(
    input_set,
    output_set
)
if network_io is not None:
    with open('data/optimized_network_cancer.h5', 'wb') as out:
        out.write(network_io.read())

#import h5py
#from keras.models import load_model

#f = h5py.File(network_io)
#model = load_model(f)

# print(model)
