# black_fox_client.OptimizationApi

All URIs are relative to *https://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_status_async**](OptimizationApi.md#get_status_async) | **GET** /api/Optimization/keras/{id}/status | 
[**post_action_async**](OptimizationApi.md#post_action_async) | **POST** /api/Optimization/keras/{id}/action/{optimizationAction} | 
[**post_async**](OptimizationApi.md#post_async) | **POST** /api/Optimization/keras | 


# **get_status_async**
> KerasOptimizationStatus get_status_async(id)



### Example
```python
from __future__ import print_function
import time
import black_fox_client
from black_fox_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = black_fox_client.OptimizationApi()
id = 'id_example' # str | 

try:
    api_response = api_instance.get_status_async(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling OptimizationApi->get_status_async: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**str**](.md)|  | 

### Return type

[**KerasOptimizationStatus**](KerasOptimizationStatus.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: text/plain, application/json, text/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **post_action_async**
> post_action_async(id, optimization_action)



### Example
```python
from __future__ import print_function
import time
import black_fox_client
from black_fox_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = black_fox_client.OptimizationApi()
id = 'id_example' # str | 
optimization_action = 'optimization_action_example' # str | 

try:
    api_instance.post_action_async(id, optimization_action)
except ApiException as e:
    print("Exception when calling OptimizationApi->post_action_async: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**str**](.md)|  | 
 **optimization_action** | **str**|  | 

### Return type

void (empty response body)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **post_async**
> str post_async(config=config)



### Example
```python
from __future__ import print_function
import time
import black_fox_client
from black_fox_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = black_fox_client.OptimizationApi()
config = black_fox_client.KerasOptimizationConfig() # KerasOptimizationConfig |  (optional)

try:
    api_response = api_instance.post_async(config=config)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling OptimizationApi->post_async: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **config** | [**KerasOptimizationConfig**](KerasOptimizationConfig.md)|  | [optional] 

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json-patch+json, application/json, text/json, application/*+json
 - **Accept**: text/plain, application/json, text/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

