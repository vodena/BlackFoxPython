from blackfox import BlackFox, KerasOptimizationConfig
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

c = KerasOptimizationConfig(validation_split=0.2)

# Use CTRL + C to stop optimization
(ann_io, ann_info, ann_metadata) = bf.optimize_keras_sync(
    input_set,
    output_set,
    config=c,
    network_path='data/optimized_network_cancer.onnx',
    network_type='onnx',
    integrate_scaler=True
)

print('\n\nann info:')
print(ann_info)

print('\n\nann metadata:')
print(ann_metadata)