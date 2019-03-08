from blackfox import BlackFox
from blackfox import PredictionFileConfig
from blackfox import Range

blackfox_url = 'http://localhost:50476/'
bf = BlackFox(blackfox_url)

config = PredictionFileConfig(
    input_ranges=[
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1)
    ],
    output_ranges=[
        Range(min=0, max=1),
        Range(min=0, max=1)
    ]
)

bf.predict_from_file_keras(
    config,
    'data/optimized_network_cancer.h5',
    'data/cancer_test_set_input.csv',
    'data/cancer_predict.csv'
)
