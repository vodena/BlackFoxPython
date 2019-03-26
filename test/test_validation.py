from __future__ import absolute_import

import unittest

from blackfox.models.keras_optimization_config import KerasOptimizationConfig
from blackfox.models.optimization_engine_config import OptimizationEngineConfig
from blackfox.models.prediction_array_config import PredictionArrayConfig
from blackfox.models.prediction_array_config import PredictionArrayConfig
from blackfox.models.keras_layer_config import KerasLayerConfig
from blackfox.models.keras_hidden_layer_config import KerasHiddenLayerConfig
from blackfox.models.prediction_file_config import PredictionFileConfig
from blackfox.models.keras_training_config import KerasTrainingConfig
from blackfox.models.range import Range
from blackfox.validation import *
from blackfox.configuration import Configuration
from blackfox.rest import ApiException


class TestValidation(unittest.TestCase):

    def tearDown(self):
        pass

    def test_validate_train_keras(self):
        input_ranges = [
            Range(min=0, max=1),
            Range(min=0, max=1),
            Range(min=0, max=1),
            Range(min=0, max=1),
            Range(min=0, max=1),
            Range(min=0, max=1),
            Range(min=0, max=1),
            Range(min=0, max=1)
        ]

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
            dropout=0,
            batch_size=100,
            dataset_id="string",
            input_ranges=input_ranges,
            output_layer=output_layer,
            hidden_layer_configs=hidden_layer_configs,
            training_algorithm='Nadam',
            max_epoch=3000,
            cross_validation=False,
            validation_split=1.7,
            random_seed=500
        )
        pass

    def test_validate_predict_file_keras(self):
        config = PredictionFileConfig(
            input_ranges=[
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
        validate_predict_from_file_keras(config)
        pass

    def test_validate_predict_array_keras(self):
        config = PredictionArrayConfig(
            input_ranges=[
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
            data_set=[
                [0.50, 1.00],
                [0.00, 0.90],
                [0.85, 0.50],
                [0.05, 0.70],
                [0.12, 1.00],
                [0.74, 0.05],
                [0.66, 0.39],
                [0.28, 0.11]
            ]
        )
        validate_test_predict_array_keras(config)
        pass

    def test_validate_optimize_keras(self):
        engine_config = OptimizationEngineConfig(
            crossover_distribution_index=20,
            crossover_probability=0.9,
            mutation_distribution_index=20,
            mutation_probability=0.01,
            proc_timeout_miliseconds=200000,
            max_num_of_generations=50,
            population_size=100,

        )

        config = KerasOptimizationConfig(
            dropout=Range(min=0, max=25),
            dataset_id='f56e2c4fa71050ee4f55c6335947ad0b9bd47d85',
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
            max_epoch=3000,
            cross_validation=False,
            validation_split=0.7,
            random_seed=100,
            engine_config=engine_config
        )

        validate_optimize_keras(config)

        pass


if __name__ == '__main__':
    unittest.main()
