# KerasSeriesOptimizationConfig

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**input_window_range_configs** | [**list[InputWindowRangeConfig]**](InputWindowRangeConfig.md) |  | [optional] 
**output_window_configs** | [**list[OutputWindowConfig]**](OutputWindowConfig.md) |  | [optional] 
**output_sample_step** | **int** |  | [optional] 
**dropout** | [**Range**](Range.md) |  | [optional] 
**batch_size** | **int** |  | [optional] 
**dataset_id** | **str** |  | [optional] 
**inputs** | [**list[InputConfig]**](InputConfig.md) |  | [optional] 
**output_ranges** | [**list[Range]**](Range.md) |  | [optional] 
**problem_type** | **str** |  | [optional] 
**hidden_layer_count_range** | [**Range**](Range.md) |  | [optional] 
**neurons_per_layer** | [**Range**](Range.md) |  | [optional] 
**training_algorithms** | **list[str]** |  | [optional] 
**activation_functions** | **list[str]** |  | [optional] 
**max_epoch** | **int** |  | 
**cross_validation** | **bool** |  | [optional] 
**validation_split** | **float** |  | 
**random_seed** | **int** |  | [optional] 
**engine_config** | [**OptimizationEngineConfig**](OptimizationEngineConfig.md) |  | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


