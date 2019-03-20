from blackfox import BlackFox
from blackfox import KerasOptimizationConfig
import h5py
from keras.models import load_model
import numpy as np
import pandas as pd

blackfox_url = 'http://localhost:50476/'
bf = BlackFox(blackfox_url)

input_count = 9
input_set = []
output_set = []

data_set = pd.read_csv('data/cancer_training_set.csv')
input_set = data_set.iloc[:, 0:input_count].values
output_set = data_set.iloc[:, input_count:].values

c = KerasOptimizationConfig(validation_split=0.2)

# Use CTRL + C to stop optimization
(ann_io, ann_status, ann_metadata) = bf.optimize_keras_sync(
    input_set,
    output_set,
    config=c,
    integrate_scaler=True
)

print('\nann info:')
print(ann_status)

print('\nann metadata:')
print(ann_metadata)

f = h5py.File(ann_io)
model = load_model(f)

#test set
test_set = pd.read_csv('data/cancer_test_set.csv')
test_input = test_set.iloc[:, 0:input_count].values

predicted = model.predict(test_input)
print(predicted)

