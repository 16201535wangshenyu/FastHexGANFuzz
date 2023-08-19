import sys
import os
from functools import cmp_to_key
import datetime
import time
import pandas as pd


def get_cov_folder_list(proto_folder):
    """
        proto_folder = {
            "coap": "coap_folder",
            "modbus": "modbus_folder",
            "mqtt": "mqtt_folder"
        }
    """
    cov_folder_list = {}
    for key, value in proto_folder.items():
        tool_cov = {}
        tools_list = os.listdir(value)
        for tool in tools_list:
            tool_path = value + os.path.sep + tool
            tool_cov[tool] = [tool_path + os.path.sep + key + "_out" + os.path.sep + "cov",tool_path + os.path.sep + key + "_out" + os.path.sep + "fuzzer_stats"]
        cov_folder_list[key] = tool_cov
    return cov_folder_list


def sort_rule(str1, str2):
    date1 = str1.split("#")
    date2 = str2.split("#")
    if date1[1] > date2[1]:
        return 1
    elif date1[1] == date2[1]:
        if date1[0] > date2[0]:
            return 1
        elif date1[0] <= date2[0]:
            return -1
    else:
        return -1


def get_time_stamp(valid_time):
    '''
    @description: 获取时间的时间戳进行时间大小比较
    @param {*} valid_time 传入的时间格式：指定为 "10/03/2023 07:56:48"
    @return {*}    返回 valid_time 的时间戳
    @author: wanghao
    '''
    dd = datetime.datetime.strptime(valid_time, '%Y-%m-%d %H.%M.%S').strftime('%Y-%m-%d %H:%M:%S')
    ts = int(time.mktime(time.strptime(dd, '%Y-%m-%d %H:%M:%S')))
    return ts


def get_time_info_by_cov_path(cov_folder_path):
    cov_folder_path_list = os.listdir(cov_folder_path[0])
    cov_folder_path_list = sorted(cov_folder_path_list, key=cmp_to_key(sort_rule))

    start_time_str = ""
    with open(cov_folder_path[1],"r") as rf:
        line = rf.readline()
        start_time_str = line
    end_time_str = cov_folder_path_list[-1].split("#")
    start_time = int(start_time_str.split(":")[1].strip())
    end_time = get_time_stamp(end_time_str[1] + " " + end_time_str[0])
    # end_time = get_time_stamp(end_time_str[1] + " " + end_time_str[0])

    return (end_time - start_time) / 60 / 60


def cal_time_by_cov_folder_list(cov_folder_list):
    """
        cov_folder_list = {
            "coap": {
                "afl++" : "cov_path",
                "aflnet" : "cov_path",
                "polar": "cov_path",
                "multifuzz" : "cov_path"
            },
            "mqtt":{……},
            "modbus":{……}
        }
    """
    time_info = {}
    for k, v in cov_folder_list.items():
        tools_time_info = {}
        for k2, v2 in v.items():
            tools_time_info[k2] = get_time_info_by_cov_path(v2)
        time_info[k] = tools_time_info
    return time_info


def pd_to_excel(data, filename):  # pandas库储存数据到excel
    proto_data = []
    tool_data = []
    time_data = []
    for k, v in data.items():
        # tool_data = tool_data + list(v.keys())
        # tool_data.append()
        for k2, time_info in v.items():
            proto_data.append(k)
            tool_data.append(k2)
            time_data.append(time_info)

    # tool_data = tool_data * 10

    dfData = {  # 用字典设置DataFrame所需数据
        'protocol': proto_data,
        'tool': tool_data,
        'time{/h}': time_data,
    }
    df = pd.DataFrame(dfData)  # 创建DataFrame
    df.to_excel(filename, index=False)  # 存表，去除原始索引列（0,1,2...）


if __name__ == "__main__":
    proto_folder = {
        "coap": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\coap",
        "modbus": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\modbus",
        "mqtt": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\mqtt"
    }
    cov_folder_list = get_cov_folder_list(proto_folder)
    time_info = cal_time_by_cov_folder_list(cov_folder_list)
    pd_to_excel(time_info, r"time.xlsx")

