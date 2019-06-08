# KerasSeriesTrainingConfig

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**input_window_configs** | [**list[InputWindowConfig]**](InputWindowConfig.md) |  | [optional] 
**output_window_configs** | [**list[OutputWindowConfig]**](OutputWindowConfig.md) |  | [optional] 
**output_sample_step** | **int** |  | [optional] 
**batch_size** | **int** |  | [optional] 
**dataset_id** | **str** |  | [optional] 
**input_ranges** | [**list[Range]**](Range.md) |  | [optional] 
**output_layer** | [**KerasLayerConfig**](KerasLayerConfig.md) |  | [optional] 
**hidden_layer_configs** | [**list[KerasHiddenLayerConfig]**](KerasHiddenLayerConfig.md) |  | [optional] 
**training_algorithm** | **str** |  | [optional] 
**max_epoch** | **int** |  | 
**cross_validation** | **bool** |  | [optional] 
**validation_split** | **float** |  | 
**random_seed** | **int** |  | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


