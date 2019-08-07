.. BlackFox documentation master file, created by
   sphinx-quickstart on Tue Aug  6 12:44:12 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Black Fox documentation
=======================

**The goal of Black Fox is to dramatically simplify
the process of building deployment-ready predictive models.**

Black Fox is a cloud-based “one click data-in model-out” 
robust artificial intelligence software solution for automated 
generation of the most adequate predictive model for any given dataset. 
It optimizes all parameters and hyperparameters of 
neural networks by genetic algorithm.
Optimization targets include number of hidden layers, 
number of neurons per layer, 
activation functions, dropout, learning algorithms, etc.
During an evolutionary process, Black Fox automatically performs adjustments of NN architecture according to the present dataset.

.. toctree::
    :maxdepth: 2

User's Guide
------------

Installation
~~~~~~~~~~~~
To install Black Fox use `pip <https://pip.pypa.io/en/stable/quickstart/>`_ or `pipenv <https://docs.pipenv.org/en/latest/>`_:

.. code-block:: PowerShell

    $ pip install -U blackfox

Example usage
~~~~~~~~~~~~~

Black Fox NN optimization without any additional parameters save the dataset requires only one function call:

.. code-block:: python

    from blackfox import BlackFox
    import csv

    input_columns = 9
    input_set = []
    output_set = []

    #Example reading and parsing .csv file 
    #with input and output set filled respectively
    with open('..path/to/your/dataset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print('Column names are ' + (", ".join(row)))
        else:
            data = list(map(float, row))
            input_set.append(data[:input_columns])
            output_set.append(data[input_columns:])

        line_count += 1

    #Create an instance of the Black Fox class by supplying api URL
    bf = BlackFox('bf.endpoint.api.address')

    #Call optimization function which returns three entities 
    #NN bytearray/model, NN info and NN JSON metadata
    (nn_io, nn_info, nn_md) = bf.optimize_keras_sync(input_set, output_set)

Optimization parameters, as well as parameters for the NN can be configured 
if needed, since the optimization method accepts several additional configuration parameters. For example:

.. code-block:: python

    from blackfox import BlackFox
    from blackfox import KerasOptimizationConfig, OptimizationEngineConfig

    #Input and output set read/parse
    ...
    ...

    #Create instances of the optimization configuration classes 
    ec = OptimizationEngineConfig(proc_timeout_seconds = 20000,
                                  population_size = 50,
                                  max_num_of_generations = 20,
                                  mutation_probability = 0.2,
                                  optimization_algorithm = "VidnerovaNeruda")

    c = KerasOptimizationConfig(engine_config = ec,
                                max_epoch = 3000,
                                validation_split = 0.2,
                                problem_type = "BinaryClassification")

    (nn_io, nn_info, nn_md) = bf.optimize_keras_sync(
      input_set, 
      output_set, 
      config = c)

Once invoked, optimization process proceeds until the most optimal NN is 
created and ultimately returned to the user (*nn_io* in the example above).
After the process, the optimized NN model can be used for prediction, either
directly or by saving the model as an *h5* file for susequent use. For
example:

.. code-block:: python

    import pandas as pd
    import h5py
    from blackfox_extras import prepare_input_data, prepare_output_data
    from keras.models import load_model

    #Optimization performed as in the first example 
    #(got nn_io, nn_info, nn_md)
    ...
    ...

    #Load and prepare test data 
    #(using functions from the blackfox_extras package)
    test_set = pd.read_csv('...path/to/test/dataset.csv')
    test_set_prepared = prepare_input_data(test_set, nn_md)

    #Save model for subsequent use
    nn_h5 = '..path/to/network.h5'
    if ann_io is not None:
    with open(nn_h5, 'wb') as out:
        out.write(ann_io.read())

    #Use model directly (or loaded from an .h5)
    f = h5py.File(nn_io)
    model = load_model(f) #Directly
    # model = load_model(nn_h5) #from .h5 file

    #Perform prediction and obtain real values
    prediction = model.predict(test_set_prepared)
    prediction_real_values = prepare_output_data(prediction, nn_md)

Please note that NN metadata can also be loaded from an *h5* file 
using *get_metadata* Black Fox method (*nn_md = bf.get_metadata('..path/to/network.h5')*).

API Guide
---------
Module
~~~~~~~~~~~~~~~
.. automodule:: blackfox
    :members:

Classes
~~~~~~~
BlackFox
++++++++
.. autoclass:: BlackFox
    :members:

LogWriter
+++++++++
.. autoclass:: LogWriter
    :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
