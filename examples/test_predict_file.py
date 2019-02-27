from blackfox import BlackFox
from blackfox import PredictionFileConfig
from blackfox import Range

bf = BlackFox('http://147.91.204.14:32700')

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
    'data/optimized_network_cancer.onnx',
    'data/ulazni_podaci_cancer_za_predikciju.csv',
    'data/rezultati_cancer.csv'
)
