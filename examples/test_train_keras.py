from blackfox import BlackFox
from blackfox import Range
from blackfox import KerasLayerConfig
from blackfox import KerasTrainingConfig
from blackfox import KerasHiddenLayerConfig

bf = BlackFox()

input_layer = KerasLayerConfig(
    activation_function='Sigmoid',
    ranges=[
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1),
        Range(min=0, max=1)
    ]
)

output_layer = KerasLayerConfig(
    activation_function='Sigmoid',
    ranges=[
        Range(min=0, max=1),
        Range(min=0, max=1)
    ]
)

hidden_layer_configs = [
    KerasHiddenLayerConfig(
        neuron_count=6,
        activation_function='Sigmoid'
    ),
    KerasHiddenLayerConfig(
        neuron_count=6,
        activation_function='Sigmoid'
    ),
    KerasHiddenLayerConfig(
        neuron_count=6,
        activation_function='Sigmoid'
    ),
    KerasHiddenLayerConfig(
        neuron_count=6,
        activation_function='Sigmoid'
    )
]

config = KerasTrainingConfig(
    dropout=1,
    input_layer=input_layer,
    output_layer=output_layer,
    hidden_layer_configs=hidden_layer_configs,
    training_algorithm='Nadam',
    max_epoch=3000,
    cross_validation=True,
    training_ratio=0.7,
    random_seed=500
)

bf.train_keras(
    config,
    'data/cancer_train_set.csv',
    'data/trained_network_cancer.onnx'
)
