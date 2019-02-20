# blackfox.TrainingApi

All URIs are relative to *https://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**post**](TrainingApi.md#post) | **POST** /api/Training/keras | 


# **post**
> TrainedNetwork post(value=value)



### Example
```python
from __future__ import print_function
import time
import blackfox
from blackfox.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = blackfox.TrainingApi()
value = blackfox.KerasTrainingConfig() # KerasTrainingConfig |  (optional)

try:
    api_response = api_instance.post(value=value)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TrainingApi->post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **value** | [**KerasTrainingConfig**](KerasTrainingConfig.md)|  | [optional] 

### Return type

[**TrainedNetwork**](TrainedNetwork.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json-patch+json, application/json, text/json, application/*+json
 - **Accept**: text/plain, application/json, text/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

