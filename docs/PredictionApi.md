# black_fox_client.PredictionApi

All URIs are relative to *https://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**post_array**](PredictionApi.md#post_array) | **POST** /api/Prediction/keras/array | 
[**post_file**](PredictionApi.md#post_file) | **POST** /api/Prediction/keras/file | 


# **post_array**
> list[list[float]] post_array(config=config)



### Example
```python
from __future__ import print_function
import time
import black_fox_client
from black_fox_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = black_fox_client.PredictionApi()
config = black_fox_client.PredictionArrayConfig() # PredictionArrayConfig |  (optional)

try:
    api_response = api_instance.post_array(config=config)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling PredictionApi->post_array: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **config** | [**PredictionArrayConfig**](PredictionArrayConfig.md)|  | [optional] 

### Return type

**list[list[float]]**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json-patch+json, application/json, text/json, application/*+json
 - **Accept**: text/plain, application/json, text/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **post_file**
> str post_file(config=config)



### Example
```python
from __future__ import print_function
import time
import black_fox_client
from black_fox_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = black_fox_client.PredictionApi()
config = black_fox_client.PredictionFileConfig() # PredictionFileConfig |  (optional)

try:
    api_response = api_instance.post_file(config=config)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling PredictionApi->post_file: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **config** | [**PredictionFileConfig**](PredictionFileConfig.md)|  | [optional] 

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json-patch+json, application/json, text/json, application/*+json
 - **Accept**: text/plain, application/json, text/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

