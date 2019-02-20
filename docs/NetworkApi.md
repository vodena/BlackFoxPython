# blackfox.NetworkApi

All URIs are relative to *https://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get**](NetworkApi.md#get) | **GET** /api/Network/{id} | Download nework file (*.onnx)
[**head**](NetworkApi.md#head) | **HEAD** /api/Network/{id} | Check if onnx file exist
[**post**](NetworkApi.md#post) | **POST** /api/Network | Upload onnx file


# **get**
> file get(id)

Download nework file (*.onnx)

### Example
```python
from __future__ import print_function
import time
import blackfox
from blackfox.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = blackfox.NetworkApi()
id = 'id_example' # str | Nework Id

try:
    # Download nework file (*.onnx)
    api_response = api_instance.get(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling NetworkApi->get: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **str**| Nework Id | 

### Return type

[**file**](file.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/octet-stream

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **head**
> head(id)

Check if onnx file exist

### Example
```python
from __future__ import print_function
import time
import blackfox
from blackfox.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = blackfox.NetworkApi()
id = 'id_example' # str | File hash(sha1)

try:
    # Check if onnx file exist
    api_instance.head(id)
except ApiException as e:
    print("Exception when calling NetworkApi->head: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **str**| File hash(sha1) | 

### Return type

void (empty response body)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **post**
> str post(file=file)

Upload onnx file

### Example
```python
from __future__ import print_function
import time
import blackfox
from blackfox.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = blackfox.NetworkApi()
file = '/path/to/file.txt' # file |  (optional)

try:
    # Upload onnx file
    api_response = api_instance.post(file=file)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling NetworkApi->post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **file** | **file**|  | [optional] 

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: multipart/form-data
 - **Accept**: text/plain, application/json, text/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

