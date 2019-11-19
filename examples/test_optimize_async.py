from blackfox import BlackFox
from blackfox import KerasOptimizationConfig, InputConfig, Range
import time
from datetime import datetime

blackfox_url = 'localhost'
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

optimization_id = bf.optimize_keras(
    data_set_path='data/cancer_training_set.csv',
    config=config
)

running = True
while running:
    status = bf.get_optimization_status_keras(
        optimization_id,
        network_path='data/cancer_async.h5'
    )
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
