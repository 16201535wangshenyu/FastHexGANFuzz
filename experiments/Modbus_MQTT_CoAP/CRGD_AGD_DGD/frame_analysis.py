import os
import random

import verify_coap_frame_vaild_tcp_num
import verify_modbus_frame_valid_num
import verify_mqtt_frame_valid_num


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
        # tools_list = os.listdir(value)
        tools_list = ["afl++","aflnet","polar","multifuzz"]
        for tool in tools_list:
            tool_path = value + os.path.sep + tool
            tool_cov[tool] = tool_path + os.path.sep + key + "_out" + os.path.sep + "fuzzing_frame.txt"
        cov_folder_list[key] = tool_cov
    return cov_folder_list


def extract_frame_from_frame_file_list(frame_file_list, tools_seed_folder):
    """
    需要提取，不在seed列表中的前20480项
    """

    if not os.path.exists(r"frame_folder"):
        os.mkdir("frame_folder")
    frame_folder_path = os.path.abspath("frame_folder")
    for k, v in frame_file_list.items():
        tools_seeds = []
        with open(tools_seed_folder + os.path.sep + k + ".txt") as tools_seed_f:
            tools_seeds = tools_seed_f.readlines()
        if not os.path.exists(frame_folder_path + os.path.sep + k):
            os.mkdir(frame_folder_path + os.path.sep + k)
        protocol_frame_folder_path = frame_folder_path + os.path.sep + k

        for k2, v2 in v.items():
            frame_file_path = protocol_frame_folder_path + os.path.sep + k2 + ".txt"
            with open(v2, "r") as rf, open(frame_file_path, "w") as wf:
                count = 0
                rf_contents = rf.readlines()
                random.shuffle(rf_contents)
                for line in rf_contents:
                    if line not in tools_seeds:
                        wf.write(line)
                        count = count + 1
                        if count >= 30000:
                            break

                # wf.writelines([:20480])


def verify_frame_valid_num(frame_folders, is_deep_learning_baseline=False):
    filter_str = []
    if is_deep_learning_baseline:
        filter_str.append("epoch")
        filter_str.append("best_model")
    else:
        filter_str.append("")
        filter_str.append("")

    # verify_frame_folder(frame_file, filter_str)
    for k, v in frame_folders.items():
        if k == "modbus":
            verify_modbus_frame_valid_num.verify_frame_folder(v, filter_str)
        elif k == "mqtt":
            verify_mqtt_frame_valid_num.verify_frame_folder(v, filter_str)
        elif k == "coap":
            verify_coap_frame_vaild_tcp_num.verify_frame_folder(v, filter_str)


if __name__ == "__main__":
    # proto_folder = {
    #     "coap": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\coap",
    #     "modbus": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\modbus",
    #     "mqtt": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\mqtt"
    # }
    # tools_seed_folder = r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\tools_seed"
    # frame_file_list = get_cov_folder_list(proto_folder)
    # extract_frame_from_frame_file_list(frame_file_list,tools_seed_folder)

    deep_learning_baseline = {
        "blstm_dcgan": {
            "coap": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\blstm_dcgan\coap\fuzzing_data",
            "modbus": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\blstm_dcgan\modbus\fuzzing_data",
            "mqtt": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\blstm_dcgan\mqtt\fuzzing_data"
        },
        "Fast_RoPEGAN": {
            "coap": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\Fast_RoPEGAN\coap\fuzzing_data",
            "modbus": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\Fast_RoPEGAN\modbus\fuzzing_data",
            "mqtt": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\Fast_RoPEGAN\mqtt\fuzzing_data"
        },
        "lstm": {
            "coap": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\lstm\coap\fuzzing_data",
            "modbus": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\lstm\modbus\fuzzing_data",
            "mqtt": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\lstm\mqtt\fuzzing_data"
        },
        "wgan": {
            "coap": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\wgan\coap\fuzzing_data",
            "modbus": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\wgan\modbus\fuzzing_data",
            "mqtt": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\wgan\mqtt\fuzzing_data"
        },
        "seq_gan": {
            "coap": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\seq_gan\coap\fuzzing_data",
            "modbus": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\seq_gan\modbus\fuzzing_data",
            "mqtt": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\seq_gan\mqtt\fuzzing_data"
        },

    }

    # frame_folders = {
    #     "coap": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\frame_folder\coap",
    #     "modbus": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\frame_folder\modbus",
    #     "mqtt": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\frame_folder\mqtt"
    # }
    for tool, frame_folder in deep_learning_baseline.items():
        verify_frame_valid_num(frame_folder, is_deep_learning_baseline=True)
