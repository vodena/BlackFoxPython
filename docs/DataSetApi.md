# blackfox.DataSetApi

All URIs are relative to *https://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get**](DataSetApi.md#get) | **GET** /api/DataSet/{id} | Download dataset file (*.csv)
[**head**](DataSetApi.md#head) | **HEAD** /api/DataSet/{id} | Check if csv file exist
[**post**](DataSetApi.md#post) | **POST** /api/DataSet | Upload csv file


# **get**
> file get(id)

Download dataset file (*.csv)

### Example
```python
from __future__ import print_function
import time
import blackfox
from blackfox.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = blackfox.DataSetApi()
id = 'id_example' # str | Dataset Id

try:
    # Download dataset file (*.csv)
    api_response = api_instance.get(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling DataSetApi->get: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **str**| Dataset Id | 

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

Check if csv file exist

### Example
```python
from __future__ import print_function
import time
import blackfox
from blackfox.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = blackfox.DataSetApi()
id = 'id_example' # str | File hash(sha1)

try:
    # Check if csv file exist
    api_instance.head(id)
except ApiException as e:
    print("Exception when calling DataSetApi->head: %s\n" % e)
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

Upload csv file

### Example
```python
from __future__ import print_function
import time
import blackfox
from blackfox.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = blackfox.DataSetApi()
file = '/path/to/file.txt' # file |  (optional)

try:
    # Upload csv file
    api_response = api_instance.post(file=file)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling DataSetApi->post: %s\n" % e)
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

