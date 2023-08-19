import os
import pandas as pd
from bs4 import BeautifulSoup
from functools import cmp_to_key


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
            tool_cov[tool] = tool_path + os.path.sep + key + "_out" + os.path.sep + "cov"
        cov_folder_list[key] = tool_cov
    return cov_folder_list


def get_cov_value_by_cov_index_file(cov_index_file):
    file = open(cov_index_file, "rb")
    html = file.read().decode("utf-8")
    # 通过html.parser解析器把我们的HTML解析成了一棵树
    bs = BeautifulSoup(html, "html.parser")
    table = bs.select("table")[0]
    table = table.select("table")[0]
    trs = table.select("tr")[1:]
    cov_value = {}

    lines_cov_info = {}
    branch_cov_info = {}
    function_cov_info = {}
    for tr in trs:
        tds = tr.select("td")
        lines_start = False
        function_start = False
        branch_start = False

        for td in tds:
            contents = td.contents
            if len(contents) == 0:
                continue

            if lines_start:
                if "hit" not in lines_cov_info.keys():
                    lines_cov_info["hit"] = str(td.contents[0]).strip()
                elif "total" not in lines_cov_info.keys():
                    lines_cov_info["total"] = str(td.contents[0]).strip()
                else:
                    lines_cov_info["coverage"] = str(td.contents[0]).strip()
                    lines_start = False
            if function_start:
                if "hit" not in function_cov_info.keys():
                    function_cov_info["hit"] = str(td.contents[0]).strip()
                elif "total" not in function_cov_info.keys():
                    function_cov_info["total"] = str(td.contents[0]).strip()

                else:
                    function_cov_info["coverage"] = str(td.contents[0]).strip()
                    function_start = False
            if branch_start:
                if "hit" not in branch_cov_info.keys():
                    branch_cov_info["hit"] = str(td.contents[0]).strip()
                elif "total" not in branch_cov_info.keys():
                    branch_cov_info["total"] = str(td.contents[0]).strip()
                else:
                    branch_cov_info["coverage"] = str(td.contents[0]).strip()
                    branch_start = False
            if "Lines" in td.contents[0]:  # 说明接下来是
                lines_start = True
            if "Functions" in td.contents[0]:
                function_start = True
            if "Branches" in td.contents[0]:
                branch_start = True
    cov_value["Lines_cov"] = lines_cov_info
    cov_value["Functions_cov"] = function_cov_info
    cov_value["Branches_cov"] = branch_cov_info
    return cov_value


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


def get_cov_info_by_cov_path(cov_folder_path):
    cov_info_list = []  # 每3w记录一次，共记录10次
    cov_folder_path_list = os.listdir(cov_folder_path)
    cov_folder_path_list = sorted(cov_folder_path_list, key=cmp_to_key(sort_rule))
    # cov_folder_path_list.sort(sort_rule)
    for per_3w_cov_folder in cov_folder_path_list:
        per_3w_cov_folder_path = cov_folder_path + os.path.sep + per_3w_cov_folder
        index_file_path = per_3w_cov_folder_path + os.path.sep + "index.html"
        result = get_cov_value_by_cov_index_file(index_file_path)
        cov_info_list.append(result)
    return cov_info_list


def collect_cov_info(cov_folder_list):
    """
    搜集覆盖率信息，从cov文件夹中
    cov_folder_list = {
        "coap": {
            "afl++" : "cov_path",
            "aflnet" : "cov_path",
            "polar": "cov_path",
            "multifuzz":"cov_path"
        },
        "mqtt":{……},
        "modbus":{……}
    }
    return:
    {
        "coap": {
            "afl++" : [function_cov_list,branch_cov_list,line_cov_list],
            "aflnet" : [function_cov_list,branch_cov_list,line_cov_list],
            "polar" : [function_cov_list,branch_cov_list,line_cov_list],
            "multifuzz" : [function_cov_list,branch_cov_list,line_cov_list]
        },
        "mqtt":{……},
        "modbus":{……}
    }

    """
    result = {}
    for key, value in cov_folder_list.items():
        all_tool_cov_info = {}
        for key2, value2 in value.items():  # key：tool, value : cov_path
            tool_cov_info = get_cov_info_by_cov_path(value2)
            all_tool_cov_info[key2] = tool_cov_info
        result[key] = all_tool_cov_info

    return result


def pd_to_excel(data, filename):  # pandas库储存数据到excel
    proto_data = []
    tool_data = []
    epoch_data = []
    lines_hit_data = []
    lines_total_data = []
    lines_cov_data = []
    branches_hit_data = []
    branches_total_data = []
    branches_cov_data = []
    function_hit_data = []
    function_total_data = []
    function_cov_data = []
    for k, v in data.items():
        # tool_data = tool_data + list(v.keys())
        # tool_data.append()
        for k2, covs in v.items():
            for i, cov in enumerate(covs):
                epoch_data.append(i + 1)
                proto_data.append(k)
                tool_data.append(k2)

                lines_hit_data.append(cov['Lines_cov']['hit'])
                lines_total_data.append(cov['Lines_cov']['total'])
                lines_cov_data.append(cov['Lines_cov']['coverage'])

                branches_hit_data.append(cov['Branches_cov']['hit'])
                branches_total_data.append(cov['Branches_cov']['total'])
                branches_cov_data.append(cov['Branches_cov']['coverage'])

                function_hit_data.append(cov['Functions_cov']['hit'])
                function_total_data.append(cov['Functions_cov']['total'])
                function_cov_data.append(cov['Functions_cov']['coverage'])

    # tool_data = tool_data * 10

    dfData = {  # 用字典设置DataFrame所需数据
        'protocol': proto_data,
        'tool': tool_data,
        'epoch': epoch_data,
        'lines_hit': lines_hit_data,
        'lines_total': lines_total_data,
        'lines_cov': lines_cov_data,
        'branches_hit': branches_hit_data,
        'branches_total': branches_total_data,
        'branches_cov': branches_cov_data,
        'function_hit': function_hit_data,
        'function_total': function_total_data,
        'function_cov': function_cov_data

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
    result = collect_cov_info(cov_folder_list)
    print(result)
    pd_to_excel(result, r'coverage.xlsx')
    # result = get_cov_info_by_cov_path(
    #     r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\modbus\multifuzz\modbus_out\cov")
    # print(result)
