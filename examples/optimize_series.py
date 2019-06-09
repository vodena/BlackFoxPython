import time
from blackfox import BlackFox
from blackfox import KerasSeriesOptimizationConfig
from blackfox import OptimizationEngineConfig
from blackfox import Range, InputWindowRangeConfig, OutputWindowConfig
import pandas as pd

data = pd.read_csv('data/series_training_set.csv')
x_train = data.iloc[:, 0:4].values.tolist()
y_train = data.iloc[:, 4:5].values.tolist()

blackfox_url = 'http://localhost:50476/'
bf = BlackFox(blackfox_url)

input_wrc = [
    InputWindowRangeConfig(window=Range(1, 4), shift=Range(0, 2)),
    InputWindowRangeConfig(window=Range(1, 4), shift=Range(0, 2)),
    InputWindowRangeConfig(window=Range(1, 4), shift=Range(0, 2)),
    InputWindowRangeConfig(window=Range(1, 4), shift=Range(0, 2))
]

output_wc = [
    OutputWindowConfig(shift=0, window=1)
]
c = KerasSeriesOptimizationConfig(
    input_window_range_configs=input_wrc,
    output_window_configs=output_wc,
    output_sample_step=1
)

start3 = time.time()

# Use CTRL + C to stop optimization
(ann_io, ann_info, ann_metadata) = bf.optimize_series_keras_sync(
    input_set=x_train,
    output_set=y_train,
    config=c,
    network_path='series.h5'
)
