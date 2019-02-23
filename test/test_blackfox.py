# coding: utf-8

from __future__ import absolute_import

import unittest

from blackfox import BlackFox  # noqa: E501
from blackfox.models import LayerConfigKerasActivationFunction
from blackfox.models import PredictionFileConfig
from blackfox.models import PredictionArrayConfig
from blackfox.models import KerasTrainingConfig
from blackfox.models import Range
from blackfox.models import HiddenLayerConfigKerasActivationFunction
from blackfox import KerasOptimizationConfig
from blackfox import OptimizationEngineConfig


class TestDataSetApi(unittest.TestCase):
    """Blackfox unit test stubs"""

    def setUp(self):
        self.blackfox = BlackFox()  # noqa: E501

    def tearDown(self):
        pass

    def test_train_keras(self):
        input_layer = LayerConfigKerasActivationFunction(
            activation_function='Sigmoid',
            ranges=[
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

        output_layer = LayerConfigKerasActivationFunction(
            activation_function='Sigmoid',
            ranges=[
                Range(min=0, max=1),
                Range(min=0, max=1)
            ]
        )

        hidden_layer_configs = [
            HiddenLayerConfigKerasActivationFunction(
                neuron_count=6,
                activation_function='Sigmoid'
            ),
            HiddenLayerConfigKerasActivationFunction(
                neuron_count=6,
                activation_function='Sigmoid'
            ),
            HiddenLayerConfigKerasActivationFunction(
                neuron_count=6,
                activation_function='Sigmoid'
            ),
            HiddenLayerConfigKerasActivationFunction(
                neuron_count=6,
                activation_function='Sigmoid'
            )
        ]

        config = KerasTrainingConfig(
            dropout=0,
            input_layer=input_layer,
            output_layer=output_layer,
            hidden_layer_configs=hidden_layer_configs,
            training_algorithm='Nadam',
            max_epoch=3000,
            cross_validation=False,
            training_ratio=0.7,
            random_seed=500
        )

        self.blackfox.train_keras(
            config,
            'C:/Users/Korisnik/Desktop/Cancer/ulazni_podaci_cancer.csv',
            'C:/Users/Korisnik/Desktop/Cancer/trained_network.onnx'
        )

        pass

    def test_predict_file_keras(self):
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
        self.blackfox.predict_from_file_keras(
            config,
            'C:/Users/Korisnik/Desktop/Cancer/trained_network.onnx',
            'C:/Users/Korisnik/Desktop/Cancer/ulazni_podaci_cancer.csv',
            'C:/Users/Korisnik/Desktop/Cancer/rezultati.csv'
        )
        pass

    def test_predict_array_keras(self):
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
        results = self.blackfox.predict_from_array_keras(
            config,
            'C:/Users/Korisnik/Desktop/Cancer/trained_network.onnx'
        )
        print(results)

        pass

    def test_sha1(self):
        sha1 = self.blackfox.sha1(
            'C:/Users/Korisnik/Desktop/Cancer/ulazni_podaci_cancer.csv'
        )
        print(sha1)
        pass

    def test_upload_data_set(self):
        id = self.blackfox.upload_data_set(
            'C:/Users/Korisnik/Desktop/Cancer/ulazni_podaci_cancer.csv'
        )
        print(id)
        pass

    def test_download_data_set(self):
        self.blackfox.download_data_set(
            'f56e2c4fa71050ee4f55c6335947ad0b9bd47d85',
            'C:/Users/Korisnik/Desktop/Cancer/ulazni_podaci_cancer_download.csv'
        )
        pass

    def test_download_network(self):
        self.blackfox.download_network(
            'B9AB1F9129944A2B7D310B05B24C5CBA097A6A86',
            'C:/Users/Korisnik/Desktop/Cancer/network_download.csv'
        )
        pass

    def test_optimize(self):
        engine_config = OptimizationEngineConfig(
            crossover_distribution_index=20,
            crossover_probability=0.9,
            mutation_distribution_index=20,
            mutation_probability=0.01,
            proc_timeout_miliseconds=200000,
            max_num_of_generations=50,
            number_of_constraints=0,
            population_size=100,
            number_of_eval_per_request=1
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
            training_ratio=0.7,
            random_seed=100,
            engine_config=engine_config
        )

        id = self.blackfox.optimize_keras(config)
        print(id)
        pass

    def test_optimization_stop(self):
        self.blackfox.stop_optimization_keras(
            'c2db8231-da65-4821-b628-b0c8be17db7d',
            'C:/Users/Korisnik/Desktop/Cancer/net.onnx'
        )
        pass

    def test_optimization_status(self):
        res = self.blackfox.get_optimization_status_keras(
            'c2db8231-da65-4821-b628-b0c8be17db7d',
            'C:/Users/Korisnik/Desktop/Cancer/net.onnx'
        )
        print(res)


if __name__ == '__main__':
    unittest.main()
