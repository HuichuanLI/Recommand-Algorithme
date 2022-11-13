# encoding: utf-8
# @author: ZhangChaoyang
# @file: rest_api.py.py
# @time: 2022-04-28

import requests
import json


def line_1st_embed(vi, vj):
    endpoint = 'http://127.0.0.1:5000/invocations'
    data = json.dumps({"columns": ["vi", "vj"], "data": [[vi, vj]]})
    headers = {"content-type": "application/json"}
    response = requests.post(endpoint, data=data, headers=headers)
    if response.status_code == 200:
        prediction = json.loads(response.text)
        # 先通过saved_model_cli show --dir model/sdne --all找到你想要的output
        embed_i = prediction[0]['output_0']
        embed_j = prediction[0]['output_1']
        print(embed_i)
        print(embed_j)
    else:
        print(response.text)


if __name__ == "__main__":
    line_1st_embed(1, 2)
    line_1st_embed(3, 4)
    line_1st_embed(2, 4)

#  python .\engineering\rest_api.py
