from blackfox import BlackFox
from blackfox import KerasOptimizationConfig, InputConfig, Range
import time
from datetime import datetime

blackfox_url = 'http://localhost:50476/'
bf = BlackFox(blackfox_url)

config = KerasOptimizationConfig(
    inputs=[
        InputConfig(Range(min=0, max=1)),
        InputConfig(Range(min=0, max=1)),
        InputConfig(Range(min=0, max=1)),
        InputConfig(Range(min=0, max=1)),
        InputConfig(Range(min=0, max=1)),
        InputConfig(Range(min=0, max=1)),
        InputConfig(Range(min=0, max=1)),
        InputConfig(Range(min=0, max=1)),
        InputConfig(Range(min=0, max=1))
    ],
    output_ranges=[
        Range(min=0, max=1),
        Range(min=0, max=1)
    ]
)

print('upload data set')
data_set_id = bf.upload_data_set('data/cancer_training_set.csv')
config.dataset_id = data_set_id
print('data set uploaded')
print('starting optimization')
optimization_id = bf.optimization_api.post_async(config=config)
print('optimization started id: '+optimization_id)

running = True
status = None
while running:
    status = bf.optimization_api.get_status_async(optimization_id)
    running = (status.state == 'Active')
    print(("%s -> %s, "
           "Generation: %s/%s, "
           "Validation set error: %f, "
           "Training set error: %f, "
           "Epoch: %d, "
           "Optimization Id: %s") % (
        datetime.now(),
        status.state,
        status.generation,
        status.total_generations,
        status.validation_set_error,
        status.training_set_error,
        status.epoch,
        optimization_id
    ))
    time.sleep(5)  # wait 5 seconds

if status.state == 'Finished' or status.state == 'Stopped':
    if status.network is not None and status.network.id is not None:
        print("Downloading network " + status.network.id)
        bf.download_network(status.network.id, path="data/cancer_rest.h5")
