import os
from tqdm import tqdm
import math

###ERROR_CODE######
OK_ERROR = 0
LEN_ERROR = -1
FIELD_ERROR = -2

num_correct_flags = [0 for r in range(100)]
num_types_of_num_correct_flags = {
    "trac_id": set(),
    "proto_id": set(),
    "data_len": set(),
    "unit_id": set(),
    "function_code": set(),
    "read_address": set(),
    "write_address": set(),
    "read_num": set(),
    "write_num": set(),
    "bytes": set(),
    "payload": set()

}


def verify_trac_id(frame):
    if frame is not None and len(frame) >= 4:
        # frame = frame[4:]
        return {
            "result": OK_ERROR
        }
    else:
        return {
            "result": LEN_ERROR
        }


def verify_proto_id(frame):
    if frame is not None and len(frame) >= 4:
        if frame[:4] == "0000":
            return {
                "result": OK_ERROR
            }
        else:
            return {
                "result": FIELD_ERROR
            }
    else:
        return {
            "result": LEN_ERROR
        }


def verify_data_len(frame):
    if frame is not None and len(frame) >= 4:
        if int("0x" + frame[:4], 16) * 2 + 4 == len(frame):
            return {
                "result": OK_ERROR
            }
        else:
            return {
                "result": LEN_ERROR
            }
    else:
        return {
            "result": LEN_ERROR
        }


def verify_unit_id(frame):
    if frame is not None and len(frame) >= 2:
        if (frame[0:2]).lower() == "ff":
            return {
                "result": OK_ERROR
            }
        else:
            return {
                "result": FIELD_ERROR
            }
    else:
        return {
            "result": LEN_ERROR
        }


"""
返回code
"""


def verify_func_code(frame):
    func_code_list = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x0f, 0x10, 0x11, 0x16, 0x17]
    if frame is not None and len(frame) >= 2:
        if int("0x" + frame[:2], 16) in func_code_list:
            return {
                "result": OK_ERROR,
                "code": (frame[:2]).lower() + "_code"
            }
        else:
            return {
                "result": FIELD_ERROR
            }
    else:
        return {
            "result": LEN_ERROR

        }


'''
起始地址、NUM，针对不同modbus实现，其起始地址都不固定，因此这里不强制限制其起始地址以及NUM
起始地址start_address：0-65535
num：0-65535 
限制 ： start_address+num<=65536
'''


def verify_01_data(frame, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    if len(frame) >= 4:  # 代表有地址字段，如果有，那么地址字段就是正确的
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["read_address"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
    if frame is not None and len(frame) == 8:
        if int("0x" + frame[:4], 16) + int("0x" + frame[4:8], 16) <= 65536 and int("0x" + frame[4:8],
                                                                                   16) != 0:  # 代表num正确
            num_correct_flags[correct_flags_num] = num_correct_flags[
                                                       correct_flags_num] + 1  # 不去计较start_address错误，而num正确的情况，
            num_types_of_num_correct_flags["read_num"].add(frame[4:8])
            correct_flags_num = correct_flags_num + 1  # start_address错误，num一定错误
            return {
                "result": OK_ERROR,
                "correct_flags_num": correct_flags_num
            }
        else:
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num
            }
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_02_data(frame, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    if len(frame) >= 4:  # 代表有地址字段，如果有，那么地址字段就是正确的
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["read_address"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
    if frame is not None and len(frame) == 8:
        if int("0x" + frame[:4], 16) + int("0x" + frame[4:8], 16) <= 65536 and int("0x" + frame[4:8], 16) != 0:
            num_correct_flags[correct_flags_num] = num_correct_flags[
                                                       correct_flags_num] + 1  # 不去计较start_address错误，而num正确的情况，
            num_types_of_num_correct_flags["read_num"].add(frame[4:8])
            correct_flags_num = correct_flags_num + 1  # start_address错误，num一定错误
            return {
                "result": OK_ERROR,
                "correct_flags_num": correct_flags_num
            }
        else:
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num
            }
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_03_data(frame, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    if len(frame) >= 4:  # 代表有地址字段，如果有，那么地址字段就是正确的
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["read_address"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
    if frame is not None and len(frame) == 8:
        if int("0x" + frame[:4], 16) + int("0x" + frame[4:8], 16) <= 65536 and int("0x" + frame[4:8], 16) != 0:  #
            num_correct_flags[correct_flags_num] = num_correct_flags[
                                                       correct_flags_num] + 1  # 不去计较start_address错误，而num正确的情况，
            num_types_of_num_correct_flags["read_num"].add(frame[4:8])
            correct_flags_num = correct_flags_num + 1  # start_address错误，num一定错误
            return {
                "result": OK_ERROR,
                "correct_flags_num": correct_flags_num
            }
        else:
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num
            }
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_04_data(frame, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    if len(frame) >= 4:  # 代表有地址字段，如果有，那么地址字段就是正确的
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["read_address"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
    if frame is not None and len(frame) == 8:
        if int("0x" + frame[:4], 16) + int("0x" + frame[4:8], 16) <= 65536 and int("0x" + frame[4:8], 16) != 0:
            num_correct_flags[correct_flags_num] = num_correct_flags[
                                                       correct_flags_num] + 1  # 不去计较start_address错误，而num正确的情况，
            num_types_of_num_correct_flags["read_num"].add(frame[4:8])
            correct_flags_num = correct_flags_num + 1  # start_address错误，num一定错误
            return {
                "result": OK_ERROR,
                "correct_flags_num": correct_flags_num
            }
        else:
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num
            }
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_05_data(frame, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    if len(frame) >= 4:  # 代表有地址字段，如果有，那么地址字段就是正确的
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["write_address"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
    if frame is not None and len(frame) == 8:
        #
        if int("0x" + frame[4:8], 16) == 0xFF00 or int("0x" + frame[4:8], 16) == 0x0000:  # 数据字段正确
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["payload"].add(frame[4:8])
            correct_flags_num = correct_flags_num + 1
            return {
                "result": OK_ERROR,
                "correct_flags_num": correct_flags_num
            }
        else:
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num
            }
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_06_data(frame, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    if len(frame) >= 4:  # 代表有地址字段，如果有，那么地址字段就是正确的
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["write_address"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
    if frame is not None and len(frame) == 8:  # 代表数据字段正确
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["payload"].add(frame[4:8])
        correct_flags_num = correct_flags_num + 1
        return {
            "result": OK_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_07_data(frame, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    if frame is not None and len(frame) == 0:
        return {
            "result": OK_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_0f_data(frame, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    if len(frame) >= 4:  # 代表有地址字段，如果有，那么地址字段就是正确的
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["write_address"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
    if frame is not None and len(frame) >= 10:

        # 255（字节） * 8（1个字节8位） 是所能表示的最多的线圈数，一位表示一个线圈值
        if int("0x" + frame[:4], 16) + int("0x" + frame[4:8], 16) <= 65536 and int("0x" + frame[4:8], 16) != 0 and int(
                "0x" + frame[4:8], 16) <= 255 * 8:  # 代表num字段正确
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["write_num"].add(frame[4:8])
            correct_flags_num = correct_flags_num + 1
            # 判断字节数是否符合规则
            if int("0x" + frame[8:10], 16) == math.ceil(int("0x" + frame[4:8], 16) / 8):  # 字节数字段正确
                num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
                num_types_of_num_correct_flags["bytes"].add(frame[8:10])
                correct_flags_num = correct_flags_num + 1
                # 判断字节数后面的数据是否合法
                if (len(frame) - 4 - 4 - 2) == int("0x" + frame[8:10], 16) * 2:  # 数据字段是否正确
                    num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
                    num_types_of_num_correct_flags["payload"].add(frame[10:])
                    correct_flags_num = correct_flags_num + 1
                    return {
                        "result": OK_ERROR,
                        "correct_flags_num": correct_flags_num
                    }
                else:
                    return {
                        "result": LEN_ERROR,
                        "correct_flags_num": correct_flags_num
                    }
            else:
                return {
                    "result": FIELD_ERROR,
                    "correct_flags_num": correct_flags_num
                }
        else:
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num
            }
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_10_data(frame, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    if len(frame) >= 4:  # 代表有地址字段，如果有，那么地址字段就是正确的
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["write_address"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
    if frame is not None and len(frame) >= 10:

        # 255（字节） // 2（1个寄存器使用2个字节表示）
        if int("0x" + frame[:4], 16) + int("0x" + frame[4:8], 16) <= 65536 and int("0x" + frame[4:8], 16) != 0 and int(
                # 判断数量字段是否正确
                "0x" + frame[4:8], 16) <= 255 // 2:
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["write_num"].add(frame[4:8])
            correct_flags_num = correct_flags_num + 1
            # 判断字节数是否符合规则
            if int("0x" + frame[8:10], 16) == int("0x" + frame[4:8], 16) * 2:
                num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
                num_types_of_num_correct_flags["bytes"].add(frame[8:10])
                correct_flags_num = correct_flags_num + 1
                # 判断字节数后面的数据是否合法
                if (len(frame) - 4 - 4 - 2) == int("0x" + frame[8:10], 16) * 2:
                    num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
                    num_types_of_num_correct_flags["payload"].add(frame[10:])
                    correct_flags_num = correct_flags_num + 1
                    return {
                        "result": OK_ERROR,
                        "correct_flags_num": correct_flags_num
                    }
                else:
                    return {
                        "result": LEN_ERROR,
                        "correct_flags_num": correct_flags_num
                    }
            else:
                return {
                    "result": FIELD_ERROR,
                    "correct_flags_num": correct_flags_num
                }
        else:
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num
            }
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_11_data(frame, correct_flags_num):
    global num_correct_flags
    if frame is not None and len(frame) == 0:
        return {
            "result": OK_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_16_data(frame, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    if len(frame) >= 4:  # 代表有地址字段，如果有，那么地址字段就是正确的
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["write_address"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
    if frame is not None and len(frame) == 12:
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["payload"].add(frame[4:12])
        correct_flags_num = correct_flags_num + 1
        return {
            "result": OK_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_17_data(frame, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    if len(frame) >= 4:  # 代表有读的起始地址字段，如果有，那么地址字段就是正确的
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["read_address"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
    if len(frame) >= 12:  # 代表有写的起始地址字段，如果有，那么地址字段就是正确的
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["write_address"].add(frame[8:12])
        correct_flags_num = correct_flags_num + 1
    read_num_flag_valid = None
    write_num_flag_valid = None
    if len(frame) >= 8:
        read_num_flag_valid = int("0x" + frame[:4], 16) + int("0x" + frame[4:8], 16) <= 65536 and int("0x" + frame[4:8],
                                                                                                      16) != 0
        if read_num_flag_valid:
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["read_num"].add(frame[4:8])
            correct_flags_num = correct_flags_num + 1

    if len(frame) >= 16:
        write_num_flag_valid = int("0x" + frame[8:12], 16) + int("0x" + frame[12:16], 16) <= 65536 and int(
            "0x" + frame[12:16], 16) != 0 and int("0x" + frame[12:16], 16) <= 255 // 2
        if write_num_flag_valid:
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["write_num"].add(frame[12:16])
            correct_flags_num = correct_flags_num + 1

    if frame is not None and len(frame) >= 18:
        # 255（字节） // 2（1个寄存器使用2个字节表示）

        # if int("0x" + frame[:4], 16) + int("0x" + frame[4:8], 16) <= 65536 and \
        #         int("0x" + frame[8:12], 16) + int("0x" + frame[12:16], 16) <= 65536 \
        #         and int("0x" + frame[4:8], 16) != 0 \
        #         and int("0x" + frame[12:16], 16) != 0 \
        #         and int("0x" + frame[12:16], 16) <= 255 // 2: # 判断

        if read_num_flag_valid and write_num_flag_valid:
            # 判断字节数是否符合规则
            if int("0x" + frame[16:18], 16) == int("0x" + frame[12:16], 16) * 2:  # 字节数合法
                num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
                num_types_of_num_correct_flags["bytes"].add(frame[16:18])
                correct_flags_num = correct_flags_num + 1
                # 判断字节数后面的数据是否合法
                if (len(frame) - 4 - 4 - 4 - 4 - 2) == int("0x" + frame[16:18], 16) * 2:  # 数据合法
                    num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
                    num_types_of_num_correct_flags["payload"].add(frame[18:])
                    correct_flags_num = correct_flags_num + 1
                    return {
                        "result": OK_ERROR,
                        "correct_flags_num": correct_flags_num

                    }
                else:
                    return {
                        "result": LEN_ERROR,
                        "correct_flags_num": correct_flags_num
                    }
            else:
                return {
                    "result": FIELD_ERROR,
                    "correct_flags_num": correct_flags_num
                }
        else:
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num
            }
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_data(frame, code, correct_flags_num):
    option_func_dict = {
        "01_code": verify_01_data,
        "02_code": verify_02_data,
        "03_code": verify_03_data,
        "04_code": verify_04_data,
        "05_code": verify_05_data,
        "06_code": verify_06_data,
        "07_code": verify_07_data,
        "0f_code": verify_0f_data,
        "10_code": verify_10_data,
        "11_code": verify_11_data,
        "16_code": verify_16_data,
        "17_code": verify_17_data
    }
    if code in option_func_dict.keys():
        return option_func_dict[code](frame, correct_flags_num)
    else:
        return {

            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_frame(frame):
    global num_correct_flags
    global num_types_of_num_correct_flags
    correct_flags_num = 0
    full_correct = True
    if verify_trac_id(frame)["result"] == OK_ERROR:

        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["trac_id"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
        frame = frame[4:]
    else:  # verify_trac_id 只会返回LEN_ERROR
        return {
            "result": False,
            "correct_flags_num": correct_flags_num
        }
    verify_proto_id_result = verify_proto_id(frame)
    if verify_proto_id_result["result"] == OK_ERROR:

        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["proto_id"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
        frame = frame[4:]
    else:
        full_correct = False
        if verify_proto_id_result["result"] == LEN_ERROR:
            return {
                "result": False,
                "correct_flags_num": correct_flags_num
            }
    verify_data_len_result = verify_data_len(frame)
    if verify_data_len_result["result"] == OK_ERROR:

        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["data_len"].add(frame[0:4])
        correct_flags_num = correct_flags_num + 1
        frame = frame[4:]
    else:
        full_correct = False
        if verify_data_len_result["result"] == LEN_ERROR:
            return {
                "result": False,
                "correct_flags_num": correct_flags_num
            }

    if verify_unit_id(frame)["result"] == OK_ERROR:

        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["unit_id"].add(frame[0:2])
        correct_flags_num = correct_flags_num + 1
        frame = frame[2:]
    else:
        full_correct = False
        return {
            "result": False,
            "correct_flags_num": correct_flags_num
        }

    result = verify_func_code(frame)
    if result["result"] == OK_ERROR:
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["function_code"].add(frame[0:2])
        correct_flags_num = correct_flags_num + 1
        frame = frame[2:]
    else:
        full_correct = False
        return {
            "result": False,
            "correct_flags_num": correct_flags_num
        }
    verify_data_result = verify_data(frame, result["code"], correct_flags_num)
    if not verify_data_result["result"] == OK_ERROR:
        return {
            "result": False,
            "correct_flags_num": verify_data_result["correct_flags_num"]
        }

    return {
        "result": full_correct,
        "correct_flags_num": verify_data_result["correct_flags_num"]
    }


def verify_frame_file(frame_file):
    valid_frame = 0
    total_frame = 0
    correct_flags_num = 0
    all_valid_frame = set()
    all_frame = set()
    global num_correct_flags
    global num_types_of_num_correct_flags
    if os.path.isfile(frame_file):
        with open(frame_file, "r") as f:
            file_contents = f.readlines()
            if file_contents is not None and len(file_contents) != 0 and file_contents[-1] == "":
                file_contents = file_contents[:-1]  # 去掉文件的最后一行是空行
            total_frame = len(file_contents)
            for frame in file_contents:
                frame = frame.rstrip("\n")
                if frame != "":
                    all_frame.add(frame)
            for frame in tqdm(all_frame):
                frame = frame.rstrip("\n")
                if frame != "":
                    cur_frame = frame
                    result_verify = verify_frame(frame)
                    if result_verify["result"] == True:
                        valid_frame = valid_frame + 1
                        all_valid_frame.add(cur_frame)
                    correct_flags_num = max(correct_flags_num, result_verify["correct_flags_num"])
                    # correct_flags_num = result_verify["correct_flags_num"]

    for i in range(correct_flags_num - 1, 0, -1):  # 记录num_correct_flags
        for j in range(i):
            num_correct_flags[j] = num_correct_flags[j] - num_correct_flags[i]

    num_correct_flags_result_str = ""
    for i in range(correct_flags_num):
        num_correct_flags_result_str = num_correct_flags_result_str + (
            "The number of the correct number field for {} is :{}\n".format(i, num_correct_flags[i]))

    ####################################记录num_types_of_num_correct_flags#################################
    num_types_of_num_correct_flags_result_str = ""
    for key in num_types_of_num_correct_flags.keys():
        num_types_of_num_correct_flags_result_str = num_types_of_num_correct_flags_result_str + (
            "{} : {}\n".format(key, len(num_types_of_num_correct_flags[key])))

    num_correct_flags = [0 for r in range(200)]  # 清空num_correct_flags
    for key in num_types_of_num_correct_flags.keys():  # 清空num_types_of_num_correct_flags
        num_types_of_num_correct_flags[key].clear()
    print("file_name is:", os.path.basename(frame_file))
    print("num_correct_flags:\n", num_correct_flags_result_str)
    print("num_types_of_num_correct_flags:\n", num_types_of_num_correct_flags_result_str)
    print("total frame is: ", total_frame)
    print("total valid_frame is: ", valid_frame)
    valid_rate = 0
    if total_frame != 0:
        valid_rate = valid_frame / total_frame

    else:
        valid_rate = 0
    print("valid rate(total_valid_frame/total_frame) is :", valid_rate)
    print("no repeat frame num is:", len(all_frame))
    result_content = num_types_of_num_correct_flags_result_str + num_correct_flags_result_str + "file_name is:{}\ntotal frame is:{}\ntotal valid_frame is:{}\nvalid rate(total_valid_frame/total_frame) is :{}\nno repeat frame num is:{}\n".format(
        os.path.basename(frame_file), total_frame, valid_frame, valid_rate, len(all_frame))

    return {
        "result": result_content
    }


"""
frame_folder存放报文的文件夹
filter_str：对文件夹中的文件进行，筛选只要文件名：epoch开头的文件
"""


def verify_frame_folder(frame_folder, filter_str):
    results_content = ""
    epoch_result = {}
    best_model_result = ""
    if os.path.isdir(frame_folder):
        file_list = os.listdir(frame_folder)
        # file_list.sort(key=lambda i: int(re.match(r'(\d+)', i).group()), reverse=False)
        for file in file_list:
            if (file.startswith(filter_str[0], 0, len(file)) or file.startswith(filter_str[1], 0,
                                                                                len(file))) and file.endswith(".txt", 0,
                                                                                                              len(file)):
                # print(file)
                result = verify_frame_file(frame_folder + os.path.sep + file)
                if file.startswith(filter_str[0], 0, len(file)):
                    try:
                        epoch_result[file] = result["result"]
                    except Exception as e:
                        print(e)
                else:
                    best_model_result = result["result"]

        # epoch_result = sorted(epoch_result.items())
        for key, value in epoch_result.items():
            results_content += value
        results_content += best_model_result
        with open(frame_folder + os.path.sep + "rich_accuracy_result.txt", "w") as result_f:
            result_f.write(results_content)


if __name__ == "__main__":
    ## sub frame
    frame_file = r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\frame_folder\modbus"
    filter_str = []
    filter_str.append("")
    filter_str.append("")
    verify_frame_folder(frame_file, filter_str)
    # verify_frame_file(frame_file)

    # folder_path = r"G:\研究生\研究生\科研\小论文\FCS-21046_Proof_hi\代码2\WGAN\output\modbus\lstm"
    #
    # folder_list = os.listdir(folder_path)
    #
    # for folder in folder_list:
    #     if os.path.isdir(folder_path + os.path.sep + folder):
    #         frame_file = folder_path + os.path.sep + folder
    #         filter_str = []
    #         filter_str.append("epoch")
    #         filter_str.append("best_model")
    #         verify_frame_folder(frame_file, filter_str)
