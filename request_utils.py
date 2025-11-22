import requests
import time

def post_url_as_json(url,headers=None,data=None,params=None):
    response = requests.post(url,headers=headers,json=data,params=params)
    if response.status_code == 200:
        # 解析响应的JSON数据为Python对象
        json_data = response.json()
        if json_data["code"]=="0":
            return json_data['data']
        else:
            raise ValueError(json_data["msg"])
    else:
        raise ValueError(response.text)

def check_response(response):
    if response.status_code == 200:
        # 解析响应的JSON数据为Python对象
        json_data = response.json()
        #print(json_data)
        # Show response message
        if json_data["code"] !="0":
            print(json_data["msg"])
    else:
        print(f'response code:{response.status_code},response text:{response.text}')

def post_request_url(url,headers=None,params=None,data=None,json=None):
    retries = 0
    while retries<3:
        try:
            response = requests.post(url,headers=headers,params=params,data=data,json=json)
            check_response(response)
            return
        except Exception as e:
            print(f"Retry {retries + 1}: {e}")
            time.sleep(30)
            retries += 1
    raise Exception("Max retries exceeded")
