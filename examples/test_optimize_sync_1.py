from blackfox import BlackFox
from blackfox import KerasOptimizationConfig
from blackfox import OptimizationEngineConfig
from blackfox import Range

blackfox_url = 'http://localhost:50476/'
bf = BlackFox(blackfox_url)

# custom config
engine_config = OptimizationEngineConfig(
    crossover_distribution_index=20,
    crossover_probability=0.9,
    mutation_distribution_index=20,
    mutation_probability=0.01,
    proc_timeout_miliseconds=200000,
    max_num_of_generations=10,
    population_size=20
)

config = KerasOptimizationConfig(
    dropout=Range(min=0, max=25),
    batch_size=10,
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
    ],
    hidden_layer_count_range=Range(min=1, max=15),
    neurons_per_layer=Range(min=1, max=10),
    training_algorithms=["SGD", "RMSprop", "Adagrad",
                         "Adadelta", "Adam", "Adamax", "Nadam"],
    activation_functions=["SoftMax", "Elu", "Selu", "SoftPlus",
                          "SoftSign", "ReLu", "TanH", "Sigmoid",
                          "HardSigmoid", "Linear"],
    max_epoch=500,
    cross_validation=False,
    validation_split=0.2,
    random_seed=100,
    engine_config=engine_config
)

# Use CTRL + C to stop optimization
network_stream = bf.optimize_keras_sync(
    config=config,
    data_set_path='data/cancer_training_set.csv',
    network_path='data/trained_network_cancer.h5'
)
