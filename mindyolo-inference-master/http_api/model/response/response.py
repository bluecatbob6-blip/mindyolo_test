from typing import List

OK = 200
FAIL = 500


def ok():
    resp = dict()
    resp["code"] = OK
    resp["msg"] = "成功"
    resp["data"] = {}
    resp["success"] = True
    return resp


def ok_with_msg(msg: str):
    resp = dict()
    resp["code"] = OK
    resp["msg"] = msg
    resp["data"] = {}
    resp["success"] = True
    return resp


def ok_with_data(data_type: str, data_list: List[str]):
    resp = dict()
    resp["code"] = OK
    resp["msg"] = "成功"
    resp["data"] = {
        "type": data_type,
        "dataList": data_list
    }
    resp["success"] = True
    return resp


def fail():
    resp = dict()
    resp["code"] = FAIL
    resp["msg"] = "失败"
    resp["data"] = {}
    resp["success"] = False
    return resp


def fail_with_msg(msg: str):
    resp = dict()
    resp["code"] = FAIL
    resp["msg"] = msg
    resp["data"] = {}
    resp["success"] = False
    return resp
