from blackfox import BlackFox, PredictionArrayConfig, Range

blackfox_url = 'http://localhost:50476/'
bf = BlackFox(blackfox_url)

config = PredictionArrayConfig(
    data_set=[
        [0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1],
        [0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.3, 0.1, 0.1],
        [0.5, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1],
        [0.5, 0.4, 0.6, 0.8, 0.4, 0.1, 0.8, 1, 0.1]
    ],
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

result = bf.predict_from_array_keras(
    config,
    'data/trained_network_cancer.h5'
)

print(result)
