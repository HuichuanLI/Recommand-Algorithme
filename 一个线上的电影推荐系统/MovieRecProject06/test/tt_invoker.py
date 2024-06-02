# -*- coding: utf-8 -*-
import requests


def t0():
    response = requests.get(
        "http://127.0.0.1:5051/test2",
        params={'name': '小明'}
    )
    if response.status_code == 200:
        res = response.json()
        if res['code'] == 200:
            print(f"调用成功&执行成功:{res}")
        else:
            print(f"调用成功&执行异常:{res}")
            if res['code'] == 201:
                print("服务器执行异常，过3s重试一次!")
            elif res['code'] == 202:
                print("参数异常，重新给定有效参数后，再重试!")
            else:
                print("其它异常....")
    else:
        print(f"网络异常:{response.status_code}")


if __name__ == '__main__':
    t0()
