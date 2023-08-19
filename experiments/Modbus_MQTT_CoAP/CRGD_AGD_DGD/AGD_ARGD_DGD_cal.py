import os
import pandas as pd
from functools import cmp_to_key
def parse_rich_accuracy_result_txt(rich_accuracy_result_txt_path):
    """
    # 对于GPF来说，其只有一个result，[afl++,aflnet,polar,multifuzz]
    # 对于深度学习的baseline来说，其有10个result, [Fast_RoPEGAN,wgan,blstm_dcgan,seqgan,lstm]
    result = [
        "flag_types" : {},
        "flags_correct_num":{},
        "total_frame_num":XX,
        "valid_frame_num":XX,
        "valid_rate":XX,
        "no_repeat_frame_num":XX
    ]
    """
    result = {}
    with open(rich_accuracy_result_txt_path, "r") as r_f:
        all_content = r_f.readlines()
        stage = "flag_types"
        flag_types = {}
        flags_correct_num = {}
        total_frame_num = 0
        valid_frame_num = 0
        valid_rate = 0
        no_repeat_frame_num = 0
        filename_epoch = 0

        # cur_file_num = 0
        for line in all_content:
            if stage == "flag_types" and line.count("The number of") != 0:
                stage = "flags_correct_num"
            elif stage == "flags_correct_num" and line.count("file_name") != 0:
                stage = "summary"
            elif stage == "summary" and ("is" not in line):  # 代表下一个文件开始了
                stage = "flag_types"
                result_t = {
                    "flag_types": flag_types, "flags_correct_num": flags_correct_num,
                    "total_frame_num": total_frame_num, "valid_frame_num": valid_frame_num,
                    "valid_rate": valid_rate, "no_repeat_frame_num": no_repeat_frame_num,
                    "filename_epoch": filename_epoch
                }
                result[filename_epoch] = result_t
                ###清除状态变量##############
                flag_types = {}
                flags_correct_num = {}
                total_frame_num = 0
                valid_frame_num = 0
                valid_rate = 0
                no_repeat_frame_num = 0
                filename_epoch = 0
                ############################

                # cur_file_num = cur_file_num + 1
                # stage = "flag_types"

            if stage == "flag_types":
                line_t = line.split(":")
                flag_types[line_t[0]] = int(line_t[1])
            elif stage == "flags_correct_num":
                line = line.lstrip("The number of the correct number field for ")
                line_t = line.split("is :")
                flags_correct_num[int(line_t[0])] = int(line_t[1])
            elif stage == "summary":
                if "file_name" in line:
                    #
                    line = line.split(":")[1]
                    line = line.lstrip("epoch").split("_")[0]
                    if line.isdigit():
                        filename_epoch = int(line) // 10 - 1
                    else:  # 是字符串
                        filename_epoch = 0

                elif "total frame" in line:
                    line = line.split(":")[1]
                    total_frame_num = int(line)
                elif "total valid_frame" in line:
                    line = line.split(":")[1]
                    valid_frame_num = int(line)
                elif "valid rate" in line:
                    line = line.split(":")[1]
                    valid_rate = float(line)
                elif "no repeat frame num" in line:
                    line = line.split(":")[1]
                    no_repeat_frame_num = int(line)

        result_t = {
            "flag_types": flag_types, "flags_correct_num": flags_correct_num,
            "total_frame_num": total_frame_num, "valid_frame_num": valid_frame_num,
            "valid_rate": valid_rate, "no_repeat_frame_num": no_repeat_frame_num,
            "filename_epoch": filename_epoch
        }

        result[filename_epoch] = result_t
        # cur_file_num = cur_file_num + 1
        return result


def cal_AGD_by_someone_epoch_result(epoch_rich_accuracy_result):
    agd = 0
    for num_flags, num in epoch_rich_accuracy_result["flags_correct_num"].items():
        agd = agd + (num / epoch_rich_accuracy_result["total_frame_num"]) * num_flags
    return agd


def cal_agd_by_rich_accuracy_result_txt(rich_accuracy_result_txt_path):
    all_tools_agd_score = {}
    for tool, proto_accuracy_txt_path in rich_accuracy_result_txt_path.items():
        proto_agd_score = {}
        for proto, accuracy_txt_path in proto_accuracy_txt_path.items():
            result = parse_rich_accuracy_result_txt(accuracy_txt_path)
            one_accuracy_result_txt_agd_score_list = {}
            for epoch, epoch_rich_accuracy_result in result.items():
                one_accuracy_result_txt_agd_score_list[epoch] = cal_AGD_by_someone_epoch_result(
                    epoch_rich_accuracy_result)
            proto_agd_score[proto] = one_accuracy_result_txt_agd_score_list
        all_tools_agd_score[tool] = proto_agd_score
    return all_tools_agd_score


def find_all_rich_accuracy_result_txt_folder(all_accuracy_txt_path):
    all_tools_accuracy_txt_path = {}
    all_tools = os.listdir(all_accuracy_txt_path)
    for tool in all_tools:
        proto_accuracy_txt_path = {}
        tool_path = all_accuracy_txt_path + os.path.sep + tool
        all_proto = os.listdir(tool_path)
        for proto in all_proto:
            proto_folder = tool_path + os.path.sep + proto
            rich_accuracy_result_file = proto_folder + os.path.sep + "rich_accuracy_result.txt"
            if not os.path.exists(rich_accuracy_result_file):
                rich_accuracy_result_file = proto_folder + os.path.sep + "fuzzing_data" + os.path.sep + "rich_accuracy_result.txt"
            proto_accuracy_txt_path[proto] = rich_accuracy_result_file

        all_tools_accuracy_txt_path[tool] = proto_accuracy_txt_path
    return all_tools_accuracy_txt_path


def cal_agd_by_all_accuracy_txt_path(all_accuracy_txt_path):
    rich_accuracy_result_txt_path = find_all_rich_accuracy_result_txt_folder(all_accuracy_txt_path)
    result = cal_agd_by_rich_accuracy_result_txt(rich_accuracy_result_txt_path)
    return result


def cal_argd_by_rich_accuracy_result_txt(rich_accuracy_result_txt_path):
    all_tools_argd_score = {}
    for tool, proto_accuracy_txt_path in rich_accuracy_result_txt_path.items():
        proto_argd_score = {}
        for proto, accuracy_txt_path in proto_accuracy_txt_path.items():
            result = parse_rich_accuracy_result_txt(accuracy_txt_path)
            one_accuracy_result_txt_argd_score_list = {}
            for epoch, epoch_rich_accuracy_result in result.items():
                argd_info_dict = {
                    "argd": epoch_rich_accuracy_result["valid_rate"],
                    "no_repeat_frame_num": epoch_rich_accuracy_result["no_repeat_frame_num"],
                }
                one_accuracy_result_txt_argd_score_list[epoch] = argd_info_dict
            proto_argd_score[proto] = one_accuracy_result_txt_argd_score_list
        all_tools_argd_score[tool] = proto_argd_score
    return all_tools_argd_score


def cal_argd_by_all_accuracy_txt_path(all_accuracy_txt_path):
    rich_accuracy_result_txt_path = find_all_rich_accuracy_result_txt_folder(all_accuracy_txt_path)
    return cal_argd_by_rich_accuracy_result_txt(rich_accuracy_result_txt_path)


def get_flag_ranking_by_epoch_accuracy_list(epoch_accuracy_list):
    """
    “flag1”:{polar_0 : rank , aflnet_0: rank, ……}
    """

    all_flag_type_value_list = {}
    for epoch_accuracy in epoch_accuracy_list:
        for epoch, accuracy in epoch_accuracy.items():
            for flag, num in accuracy["flag_types"].items():
                if flag not in all_flag_type_value_list.keys():
                    all_flag_type_value_list[flag] = set()
                    all_flag_type_value_list[flag].add(num)
                else:
                    all_flag_type_value_list[flag].add(num)
    # 对 all_flag_type_value_list 中每一个flag中的list元素进行排序
    for k, v in all_flag_type_value_list.items():
        sorted_v = list(v)
        sorted_v.sort(reverse=True)
        all_flag_type_value_list[k] = sorted_v

    all_flag_ranking = {}
    for epoch_accuracy in epoch_accuracy_list:
        for epoch,accuracy in epoch_accuracy.items():
            for flag, num in accuracy["flag_types"].items():
                if flag not in all_flag_ranking.keys():
                    all_flag_ranking[flag] = {}
                    all_flag_ranking[flag][accuracy["tool_name"] + "_" + str(accuracy["filename_epoch"])] = \
                        all_flag_type_value_list[flag].index(num)
                else:
                    all_flag_ranking[flag][accuracy["tool_name"] + "_" + str(accuracy["filename_epoch"])] = \
                        all_flag_type_value_list[flag].index(num)

    return all_flag_ranking


def cal_dgd_by_one_epoch_accuracy_result_txt(proto_flag_ranking, one_epoch_accuracy_result_txt, proto, tool):
    flag_types = one_epoch_accuracy_result_txt["flag_types"]
    filename_epoch = one_epoch_accuracy_result_txt["filename_epoch"]
    dgd = 0
    for flag, type_num in flag_types.items():
        dgd = dgd + proto_flag_ranking[proto][flag][tool + "_" + str(filename_epoch)]
    return dgd


def cal_dgd_by_rich_accuracy_result_txt(rich_accuracy_result_txt_path):
    """
    给出两组结果，一种是epoch参与排名的，另一种是从10个epoch找出一个代表参与排名的
    """
    ### 计算epoch参与排名的
    proto_epoch_accuracy_list = {}
    for tool, proto_accuracy_txt_path in rich_accuracy_result_txt_path.items():
        for proto, accuracy_txt_path in proto_accuracy_txt_path.items():
            result = parse_rich_accuracy_result_txt(accuracy_txt_path)
            if proto not in proto_epoch_accuracy_list.keys():
                proto_epoch_accuracy_list[proto] = []
                for k,v in result.items():
                    result[k]["tool_name"] = tool

                proto_epoch_accuracy_list[proto].append(result)
            else:
                for k,v in result.items():
                    result[k]["tool_name"] = tool
                # result["tool_name"] = tool
                proto_epoch_accuracy_list[proto].append(result)
            # for epoch, epoch_rich_accuracy_result in result.items():
            #     pass
    proto_flag_ranking = {}
    ## 得到每一个协议，flag的排名
    for proto, epoch_accuracy_list in proto_epoch_accuracy_list.items():
        all_flag_ranking = get_flag_ranking_by_epoch_accuracy_list(epoch_accuracy_list)
        proto_flag_ranking[proto] = all_flag_ranking

    all_tools_dgd_score = {}
    for tool, proto_accuracy_txt_path in rich_accuracy_result_txt_path.items():
        proto_dgd_score = {}

        for proto, accuracy_txt_path in proto_accuracy_txt_path.items():
            result = parse_rich_accuracy_result_txt(accuracy_txt_path)
            one_accuracy_result_txt_dgd_score_list = {}
            for epoch, epoch_rich_accuracy_result in result.items():
                # 计算dgd
                dgd = cal_dgd_by_one_epoch_accuracy_result_txt(proto_flag_ranking, epoch_rich_accuracy_result, proto,
                                                               tool)
                # dgd_info_dict = {
                #     "dgd": epoch_rich_accuracy_result["valid_rate"],
                #     "no_repeat_frame_num": epoch_rich_accuracy_result["no_repeat_frame_num"],
                # }
                one_accuracy_result_txt_dgd_score_list[epoch] = dgd
            proto_dgd_score[proto] = one_accuracy_result_txt_dgd_score_list
        all_tools_dgd_score[tool] = proto_dgd_score

    return all_tools_dgd_score

    ### 先从10个epoch中找到DGD最好的一个epoch，然后以此为代表和其他的baseline比较计算DGD


##  以此计算dgd
def cal_dgd_by_rich_accuracy_result_txt_2(all_tools_dgd_score, rich_accuracy_result_txt_path):
    ### 先从10个epoch中找到DGD最好的一个epoch，然后以此为代表和其他的baseline比较计算DGD
    tools_proto_best_choose_epoch = {}
    for tools, proto_dgd_score in all_tools_dgd_score.items():
        proto_best_choose_epoch = {}
        for proto, epoch_dgd_score in proto_dgd_score.items():
            proto_best_choose_epoch[proto] = 0
            for epoch, dgd_score in epoch_dgd_score.items():
                if epoch_dgd_score[proto_best_choose_epoch[proto]] > dgd_score:
                    proto_best_choose_epoch[proto] = epoch
        tools_proto_best_choose_epoch[tools] = proto_best_choose_epoch

    ### 计算dgd

    proto_epoch_accuracy_list = {}
    for tool, proto_accuracy_txt_path in rich_accuracy_result_txt_path.items():
        for proto, accuracy_txt_path in proto_accuracy_txt_path.items():
            result = parse_rich_accuracy_result_txt(accuracy_txt_path)
            for epoch,accuracy_result in result.items():
                if accuracy_result["filename_epoch"] == tools_proto_best_choose_epoch[tool][proto]:
                    if proto not in proto_epoch_accuracy_list.keys():
                        proto_epoch_accuracy_list[proto] = []
                        result[epoch]["tool_name"] = tool
                        result_t  ={epoch:result[epoch]}
                        proto_epoch_accuracy_list[proto].append(result_t)
                    else:
                        result[epoch]["tool_name"] = tool
                        result_t = {epoch: result[epoch]}
                        proto_epoch_accuracy_list[proto].append(result_t)
                    # for epoch, epoch_rich_accuracy_result in result.items():
                    #     pass
    proto_flag_ranking = {}
    ## 得到每一个协议，flag的排名
    for proto, epoch_accuracy_list in proto_epoch_accuracy_list.items():
        all_flag_ranking = get_flag_ranking_by_epoch_accuracy_list(epoch_accuracy_list)
        proto_flag_ranking[proto] = all_flag_ranking

    all_tools_dgd_score = {}
    for tool, proto_accuracy_txt_path in rich_accuracy_result_txt_path.items():
        proto_dgd_score = {}

        for proto, accuracy_txt_path in proto_accuracy_txt_path.items():
            result = parse_rich_accuracy_result_txt(accuracy_txt_path)
            one_accuracy_result_txt_dgd_score_list = {}
            for epoch, epoch_rich_accuracy_result in result.items():
                # 计算dgd
                if result[epoch]["filename_epoch"] == tools_proto_best_choose_epoch[tool][proto]:
                    dgd = cal_dgd_by_one_epoch_accuracy_result_txt(proto_flag_ranking, epoch_rich_accuracy_result,
                                                                   proto,
                                                                   tool)

                    one_accuracy_result_txt_dgd_score_list[epoch] = dgd
            proto_dgd_score[proto] = one_accuracy_result_txt_dgd_score_list
        all_tools_dgd_score[tool] = proto_dgd_score

    return all_tools_dgd_score


def cal_dgd_by_accuracy_txt_path(all_accuracy_txt_path):
    rich_accuracy_result_txt_path = find_all_rich_accuracy_result_txt_folder(all_accuracy_txt_path)
    all_epoch_dgd = cal_dgd_by_rich_accuracy_result_txt(rich_accuracy_result_txt_path)
    best_epoch_dgd = cal_dgd_by_rich_accuracy_result_txt_2(all_epoch_dgd,rich_accuracy_result_txt_path)
    return {
        "all_epoch" : all_epoch_dgd,
        "best_epoch" : best_epoch_dgd
    }



def AGD_pd_to_excel(data, proto ,filename):  # pandas库储存数据到excel
    epoch_data = []
    dfData = {}
    for i in range(10):
        epoch_data.append(i+1)
    dfData['epoch'] = epoch_data
    # tool_data = tool_data * 10

    for tool , proto_agd in data.items():

        tmp_data = []
        if tool in ["afl++","aflnet","multifuzz","polar"]:
            tmp_data = [proto_agd[proto][0]] * 10
        else:
            tmp_data = [0] * 10
            for epoch,agd in proto_agd[proto].items():
                tmp_data[epoch] = agd
        dfData[tool] = tmp_data

    df = pd.DataFrame(dfData)  # 创建DataFrame
    df.to_excel(filename, index=False)  # 存表，去除原始索引列（0,1,2...）


def ARGD_pd_to_excel(data ,proto , filename):  # pandas库储存数据到excel
    epoch_data = []
    dfData = {}
    for i in range(10):
        epoch_data.append(i+1)
    dfData['epoch'] = epoch_data
    # tool_data = tool_data * 10
    for tool , proto_agd in data.items():

        tmp_data = []
        if tool in ["afl++","aflnet","multifuzz","polar"]:
            tmp_data = [proto_agd[proto][0]["argd"]] * 10
        else:
            tmp_data = [0] * 10
            for epoch,agd in proto_agd[proto].items():
                tmp_data[epoch] = agd["argd"]
        dfData[tool] = tmp_data

    df = pd.DataFrame(dfData)  # 创建DataFrame
    df.to_excel(filename, index=False)  # 存表，去除原始索引列（0,1,2...）

def Repeat_data_pd_to_excel(data, proto ,filename):  # pandas库储存数据到excel
    epoch_data = []
    dfData = {}
    for i in range(10):
        epoch_data.append(i+1)
    dfData['epoch'] = epoch_data
    # tool_data = tool_data * 10

    for tool , proto_agd in data.items():

        tmp_data = []
        if tool in ["afl++","aflnet","multifuzz","polar"]:
            tmp_data = [proto_agd[proto][0]["no_repeat_frame_num"]/30000] * 10
        else:
            tmp_data = [0] * 10
            for epoch,agd in proto_agd[proto].items():
                tmp_data[epoch] = agd["no_repeat_frame_num"]/30000
        dfData[tool] = tmp_data

    df = pd.DataFrame(dfData)  # 创建DataFrame
    df.to_excel(filename, index=False)  # 存表，去除原始索引列（0,1,2...）

if __name__ == "__main__":
    # cal AGD
    file = r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt"
    # agd_data = cal_agd_by_all_accuracy_txt_path(file)
    # print(agd_data)
    # AGD_pd_to_excel(agd_data,"mqtt","AGD_MQTT.xlsx")
    # AGD_pd_to_excel(agd_data, "modbus", "AGD_MODBUS.xlsx")
    # AGD_pd_to_excel(agd_data, "coap", "AGD_COAP.xlsx")

    # cal ARGD 生成数据的准确率  以及合法数据帧的不重复的种类
    # ARGD = cal_argd_by_all_accuracy_txt_path(file)
    # print(ARGD)
    # ARGD_pd_to_excel(ARGD,"mqtt","ARGD_MQTT.xlsx")
    # ARGD_pd_to_excel(ARGD, "modbus", "ARGD_MODBUS.xlsx")
    # ARGD_pd_to_excel(ARGD, "coap", "ARGD_COAP.xlsx")
    # Repeat_data_pd_to_excel(ARGD,"mqtt","Repeat_data_MQTT.xlsx")
    # Repeat_data_pd_to_excel(ARGD,"modbus", "Repeat_data_MODBUS.xlsx")
    # Repeat_data_pd_to_excel(ARGD,"coap", "Repeat_data_COAP.xlsx")

    # cal DGD
    # # 全部的epoch参与运算
    result  = cal_dgd_by_accuracy_txt_path(file)
    # # 最好的一个epoch参与运算
    print("all_epoch", result["all_epoch"])
    print("best_epoch",result["best_epoch"])



