import os
from tqdm import tqdm
import sys



UNLIMIT = sys.maxsize
###ERROR_CODE######
OK_ERROR = 0
LEN_ERROR = -1
FIELD_ERROR = -2

num_correct_flags = [0 for r in range(100)]
num_types_of_num_correct_flags = {
    "type": set(),
    "flags": set(),
    "len": set(),
    # connect报文的
    "proto_name": set(),
    "proto_level": set(),
    "connect_flags": set(),
    "keep_alive": set(),
    "client_id": set(),
    "will_topic": set(),
    "will_message": set(),
    "username": set(),
    "password": set(),
    # publish
    "topic_name": set(),
    "packet_identifier": set(),
    "payload": set(),
    # puback
    # pubrec
    # pubrel
    # pubcomp
    # subscribe
    "qos": set(),
    # unsubscribe
    # pingReq
    # disconnect

}


def get_bit_val(byte, index):
    """
    得到某个字节中某一位(Bit)的值

    :param byte: 待取值的字节值

    :param index: 待读取位的序号，从右向左0开始，0-7为一个完整字节的8个位

    :returns: 返回读取该位的值，0或1
    """
    if byte & (1 << index):

        return 1
    else:
        return 0


def verify_frame_type(frame):
    if frame is not None and len(frame) >= 1:
        if int("0x" + frame[0], 16) in [1, 3, 4, 5, 6, 7, 8, 0xa, 0xc, 0xe]:
            return {
                "result": OK_ERROR,
                "frame_type": int("0x" + frame[0], 16)
            }
        else:
            return {
                "result": FIELD_ERROR,
                "frame_type": -1
            }
    else:
        return {
            "result": LEN_ERROR
        }


def verify_frame_flags(frame, frame_type):
    if frame is not None and len(frame) >= 1:
        if (frame_type in [1, 4, 5, 7, 0xc, 0xe] and frame[0] == "0") or (frame_type in [6, 8, 0xa] and frame[0] == "2") \
                or (frame_type in [3] and frame[0] not in ["6", "7", "e", "f"]):  # not in 0110 0111 1110 1111 qos不为11
            num_flags = int("0x" + frame[0], 16)
            frame_flags = {
                "DUP": get_bit_val(num_flags, 3),
                "Qos": get_bit_val(num_flags, 2) * 2 + get_bit_val(num_flags, 1),
                "retain": get_bit_val(num_flags, 0)
            }
            return {
                "result": OK_ERROR,
                "frame_flags": frame_flags
            }
        else:
            frame_flags = {
                "DUP": -1,
                "Qos": -1,
                "retain": -1
            }
            return {
                "result": FIELD_ERROR,
                "frame_flags": frame_flags
            }
    else:
        return {
            "result": LEN_ERROR
        }


"""
判断报文的len字段是否合法
"""


def decode_len(frame):
    multiplier = 1
    value = 0
    i = 0

    while True:
        encodedByte = 0
        if len(frame) >= 2 * i + 2:
            encodedByte = int("0x" + frame[2 * i:2 * i + 2], 16)
            i = i + 1
            value = value + (encodedByte & 127) * multiplier
            multiplier = multiplier * 128
            if (multiplier > 128 * 128 * 128 * 128):
                return {  # len长度大于4个字节
                    "result": FIELD_ERROR
                }
            if (encodedByte & 128) == 0:
                break

        else:
            return {
                "result": LEN_ERROR
            }

    return {
        "result": OK_ERROR,
        "frame_len": value,
        "len_bytes_num": i
    }


def verify_len(frame):
    if frame is not None and len(frame) >= 2:
        result = decode_len(frame)
        ## 判断报文是不是还有len字节
        if result["result"] == OK_ERROR:
            if result["len_bytes_num"] * 2 + result["frame_len"] * 2 == len(frame):
                return result
            else:
                return {
                    "result": FIELD_ERROR
                }
        else:
            return result
    else:
        return {
            "result": LEN_ERROR
        }


def verify_utf8_str(frame, min_len, max_len):
    if len(frame) >= 4:
        str_len = int("0x" + frame[:4], 16)
        if (min_len <= str_len <= max_len) and len(frame) >= str_len * 2 + 4:
            return {
                "result": OK_ERROR,
                "str_len": str_len
            }
        else:
            return {
                "result": LEN_ERROR
            }


    else:
        return {
            "result": LEN_ERROR
        }


def verify_opt_head_and_payload_connect(frame, frame_flags, correct_flags_num):
    ### 验证opt head 部分 #################
    global num_correct_flags
    global num_types_of_num_correct_flags
    full_correct = OK_ERROR
    # 1、proto_name
    if len(frame) >= 12:
        if frame[0:12] != "00044d515454":  # mqtt
            full_correct = FIELD_ERROR
            # return {
            #     "result": FIELD_ERROR
            # }
        else:  # proto_name 字段正确
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["proto_name"].add(frame[0:12])
            correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }
    # 2 、proto_level 04
    if len(frame) >= 12 + 2:
        if frame[12:14] != "04":
            full_correct = FIELD_ERROR if full_correct == OK_ERROR else full_correct
            # return {
            #     "result": FIELD_ERROR
            # }
        else:
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["proto_level"].add(frame[12:14])
            correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num

        }
    # 3、Connect_Flags 1 byte
    connect_flags = {}
    if len(frame) >= 12 + 2 + 2:
        # 能够出现的组合一共42种，如果username 0 ，则password 0 ，如果will-flags 0 ，则will-qos will-retain 0
        flags_num = int("0x" + frame[14:16], 16)
        connect_flags["username"] = get_bit_val(flags_num, 7)
        connect_flags["password"] = get_bit_val(flags_num, 6)
        connect_flags["will_retain"] = get_bit_val(flags_num, 5)
        connect_flags["will_qos"] = get_bit_val(flags_num, 4) * 2 + get_bit_val(flags_num, 3)
        connect_flags["will_flag"] = get_bit_val(flags_num, 2)
        connect_flags["clean_session"] = get_bit_val(flags_num, 1)
        connect_flags["reserved"] = get_bit_val(flags_num, 0)

        if not (not (connect_flags["username"] == 0 and connect_flags["password"] == 1)) and \
                (connect_flags["will_qos"] != 3) and \
                ((connect_flags["will_flag"] == 0 and connect_flags["will_qos"] == 0 and connect_flags[
                    "will_retain"] == 0)
                 or (get_bit_val(flags_num, 2) == 1)) and connect_flags["reserved"] == 0:
            full_correct = FIELD_ERROR if full_correct == OK_ERROR else full_correct
            # return {
            #     "result": FIELD_ERROR
            # }
        else:  # Connect_Flags字段正确
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["connect_flags"].add(frame[14:16])
            correct_flags_num = correct_flags_num + 1

    else:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }
    # 4、keep_alive 2 byte
    if not len(frame) >= 12 + 2 + 2 + 4:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:  # keep_alive字段正确
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["keep_alive"].add(frame[16:20])
        correct_flags_num = correct_flags_num + 1
    ######################## payload ########################
    # 5、client_id 必须有
    client_id_result = verify_utf8_str(frame[12 + 2 + 2 + 4:], 0, 23)
    if client_id_result["result"] == OK_ERROR:
        if client_id_result["str_len"] == 0 and connect_flags["clean_session"] == 0:
            full_correct = FIELD_ERROR if full_correct == OK_ERROR else full_correct
            # return {
            #     "result": FIELD_ERROR
            # }
        else:  # client_id字段正确
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["client_id"].add(frame[20:20 + client_id_result["str_len"] * 2 + 4])
            correct_flags_num = correct_flags_num + 1
    else:  # verify_utf8_str函数只会返回LEN_ERROR || OK_ERROR因此该函数遇见错误无法往下，只能返回
        client_id_result["correct_flags_num"] = correct_flags_num
        return client_id_result

    # 6、will_topic
    will_topic_result = {"str_len": -2}
    if connect_flags["will_flag"] == 1:  # 代表有will_topic
        will_topic_result = verify_utf8_str(frame[12 + 2 + 2 + 4 + 4 + client_id_result["str_len"] * 2:], 1, UNLIMIT)
        if will_topic_result["result"] != OK_ERROR:
            will_topic_result["correct_flags_num"] = correct_flags_num
            return will_topic_result
        else:  # will_topic字段正确
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["will_topic"].add(frame[
                                                             24 + client_id_result["str_len"] * 2:24 + client_id_result[
                                                                 "str_len"] * 2 + will_topic_result["str_len"] * 2 + 4])
            correct_flags_num = correct_flags_num + 1
    # 7、will_message
    will_message_result = {"str_len": -2}
    if connect_flags["will_flag"] == 1:  # 代表有will_message
        will_message_result = verify_utf8_str(
            frame[12 + 2 + 2 + 4 + 4 + client_id_result["str_len"] * 2 + 4 + will_topic_result["str_len"] * 2:], 0,
            UNLIMIT)
        if will_message_result["result"] != OK_ERROR:
            will_message_result["correct_flags_num"] = correct_flags_num
            return will_message_result
        else:  # will_message 字段正确
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["will_message"].add(
                frame[28 + client_id_result["str_len"] * 2 + will_topic_result["str_len"] * 2:
                      28 + client_id_result["str_len"] * 2 + will_topic_result["str_len"] * 2 + 4 + will_message_result[
                          "str_len"] * 2])
            correct_flags_num = correct_flags_num + 1
    # 8、username
    username_result = {"str_len": -2}
    if connect_flags["username"] == 1:  # 代表有username
        username_result = verify_utf8_str(frame[
                                          12 + 2 + 2 + 4 + 4 + client_id_result["str_len"] * 2 + 4 + will_topic_result[
                                              "str_len"] * 2 + 4 + will_message_result["str_len"] * 2:], 1, UNLIMIT)
        if username_result["result"] != OK_ERROR:
            username_result["correct_flags_num"] = correct_flags_num
            return username_result
        else:  # username字段正确
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["username"].add(
                frame[32 + client_id_result["str_len"] * 2 + will_topic_result["str_len"] * 2 + will_message_result[
                    "str_len"] * 2:
                      32 + client_id_result["str_len"] * 2 + will_topic_result["str_len"] * 2 + will_message_result[
                          "str_len"] * 2 + 4 + username_result["str_len"] * 2])
            correct_flags_num = correct_flags_num + 1
    # 9、password
    password_result = {"str_len": -2}
    if connect_flags["password"] == 1:  # 代表有username
        password_result = verify_utf8_str(frame[
                                          12 + 2 + 2 + 4 + 4 + client_id_result["str_len"] * 2 + 4 + will_topic_result[
                                              "str_len"] * 2 + 4 + will_message_result["str_len"] * 2 + 4 +
                                          username_result["str_len"] * 2:], 0, UNLIMIT)
        if password_result["result"] != OK_ERROR:
            password_result["correct_flags_num"] = correct_flags_num
            return password_result
        else:
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["password"].add(
                frame[36 + client_id_result["str_len"] * 2 + will_topic_result["str_len"] * 2 + will_message_result[
                    "str_len"] * 2 + username_result["str_len"] * 2:
                      36 + client_id_result["str_len"] * 2 + will_topic_result["str_len"] * 2 + will_message_result[
                          "str_len"] * 2 + username_result["str_len"] * 2 + 4 + password_result["str_len"] * 2])
            correct_flags_num = correct_flags_num + 1

    return {
        "result": OK_ERROR | full_correct,
        "correct_flags_num": correct_flags_num
    }


def verify_opt_head_and_payload_publish(frame, frame_flags, correct_flags_num):
    # frame_flags = {
    #     "DUP": get_bit_val(num_flags, 3),
    #     "Qos": get_bit_val(num_flags, 2) * 2 + get_bit_val(num_flags, 1),
    #     "retain": get_bit_val(num_flags, 0)
    # }
    global num_correct_flags
    global num_types_of_num_correct_flags
    ################### opt_head ############
    # 1、topic_name
    topic_name_result = verify_utf8_str(frame, 1, UNLIMIT)
    if topic_name_result["result"] != OK_ERROR:
        topic_name_result["correct_flags_num"] = correct_flags_num
        return topic_name_result
    else:
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["topic_name"].add(frame[:topic_name_result["str_len"] * 2 + 4])
        correct_flags_num = correct_flags_num + 1
    # 2、packet_identifier
    if frame_flags["Qos"] > 0:
        if not (len(frame) >= 4 + topic_name_result["str_len"] * 2 + 4):
            return {
                "correct_flags_num": correct_flags_num,
                "result": LEN_ERROR
            }
        else:  # packet_identifier 正确
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["packet_identifier"].add(
                frame[topic_name_result["str_len"] * 2 + 4:topic_name_result["str_len"] * 2 + 4 + 4])
            correct_flags_num = correct_flags_num + 1

    ######### payload data 因为这部分data可有可无 0-N字节，需要理会 ###############payload正确
    num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
    num_types_of_num_correct_flags["payload"].add(frame[topic_name_result["str_len"] * 2 + 8:])
    correct_flags_num = correct_flags_num + 1
    return {
        "result": OK_ERROR,
        "correct_flags_num": correct_flags_num
    }


def verify_opt_head_and_payload_puback(frame, frame_flags, correct_flags_num):
    ################### opt_head ############
    # 1、packet_identifier
    # packet_identifier_len = 0
    # if frame_flags["Qos"] > 0:
    global num_correct_flags
    global num_types_of_num_correct_flags
    packet_identifier_len = 4
    if not (len(frame) >= 4):
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:  # packet_identifier正确
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["packet_identifier"].add(frame[:4])
        correct_flags_num = correct_flags_num + 1
    ################# payload 无#############
    if len(frame) > packet_identifier_len:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }

    return {
        "result": OK_ERROR,
        "correct_flags_num": correct_flags_num
    }


def verify_opt_head_and_payload_pubrec(frame, frame_flags, correct_flags_num):
    ################### opt_head ############
    # 1、packet_identifier
    # packet_identifier_len = 0
    # if frame_flags["Qos"] > 0:
    global num_correct_flags
    global num_types_of_num_correct_flags
    packet_identifier_len = 4
    if not (len(frame) >= 4):
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:  # packet_identifier正确
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["packet_identifier"].add(frame[:4])
        correct_flags_num = correct_flags_num + 1
    ################# payload 无#############
    if len(frame) > packet_identifier_len:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }

    return {
        "result": OK_ERROR,
        "correct_flags_num": correct_flags_num
    }


def verify_opt_head_and_payload_pubrel(frame, frame_flags, correct_flags_num):
    ################### opt_head ############
    # 1、packet_identifier
    # packet_identifier_len = 0
    # if frame_flags["Qos"] > 0:
    global num_correct_flags
    global num_types_of_num_correct_flags
    packet_identifier_len = 4
    if not (len(frame) >= 4):
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:  # packet_identifier正确
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["packet_identifier"].add(frame[:4])
        correct_flags_num = correct_flags_num + 1
    ################# payload 无#############
    if len(frame) > packet_identifier_len:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }

    return {
        "result": OK_ERROR,
        "correct_flags_num": correct_flags_num
    }


def verify_opt_head_and_payload_pubcomp(frame, frame_flags, correct_flags_num):
    ################### opt_head ############
    # 1、packet_identifier
    # packet_identifier_len = 0
    # if frame_flags["Qos"] > 0:
    global num_correct_flags
    global num_types_of_num_correct_flags
    packet_identifier_len = 4
    if not (len(frame) >= 4):
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:  # packet_identifier正确
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["packet_identifier"].add(frame[:4])
        correct_flags_num = correct_flags_num + 1
    ################# payload 无#############
    if len(frame) > packet_identifier_len:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }

    return {
        "result": OK_ERROR,
        "correct_flags_num": correct_flags_num
    }


def verify_opt_head_and_payload_subscribe(frame, frame_flags, correct_flags_num):
    ################### opt_head ############
    # 1、packet_identifier
    # packet_identifier_len = 0
    # if frame_flags["Qos"] > 0:
    global num_correct_flags
    global num_types_of_num_correct_flags
    full_correct = OK_ERROR
    packet_identifier_len = 4
    if not (len(frame) >= 4):
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["packet_identifier"].add(frame[:4])
        correct_flags_num = correct_flags_num + 1
    ################# payload 有#############
    i = 0
    count = 0
    while (packet_identifier_len + i) < len(frame):
        # 1、topic_name
        topic_name_result = verify_utf8_str(frame[packet_identifier_len + i:], 1, UNLIMIT)
        if topic_name_result["result"] != OK_ERROR:
            topic_name_result["correct_flags_num"] = correct_flags_num
            return topic_name_result

        else:

            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["topic_name"].add(
                frame[packet_identifier_len + i:packet_identifier_len + i + topic_name_result["str_len"] * 2 + 4])
            correct_flags_num = correct_flags_num + 1
            i = i + topic_name_result["str_len"] * 2 + 4

        # 2、qos
        if len(frame) >= 2 + packet_identifier_len + i:
            if int("0x"+ frame[packet_identifier_len + i:packet_identifier_len + i + 2],16) > 2:
                full_correct = FIELD_ERROR
                # return {
                #     "result": FIELD_ERROR
                # }
            else:  # qos正确
                count = count + 1
                num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
                num_types_of_num_correct_flags["qos"].add(
                    frame[packet_identifier_len + i:packet_identifier_len + i + 2])
                correct_flags_num = correct_flags_num + 1
                i = i + 2
        else:
            return {
                "result": LEN_ERROR,
                "correct_flags_num": correct_flags_num

            }
    if count == 0:  # 至少有一对topic_name以及qos
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num
        }

    return {
        "result": OK_ERROR | full_correct,
        "correct_flags_num": correct_flags_num
    }


def verify_opt_head_and_payload_unsubscribe(frame, frame_flags, correct_flags_num):
    ################### opt_head ############
    # 1、packet_identifier
    # packet_identifier_len = 0
    # if frame_flags["Qos"] > 0:
    global num_correct_flags
    global num_types_of_num_correct_flags

    packet_identifier_len = 4
    if not (len(frame) >= 4):
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num

        }
    else:
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["packet_identifier"].add(frame[:4])
        correct_flags_num = correct_flags_num + 1
    ################# payload 有#############
    i = 0
    count = 0
    while (packet_identifier_len + i) < len(frame):
        # 1、topic_name
        topic_name_result = verify_utf8_str(frame[packet_identifier_len + i:], 1, UNLIMIT)
        if topic_name_result["result"] != OK_ERROR:
            topic_name_result["correct_flags_num"] = correct_flags_num
            return topic_name_result
        else:
            count = count + 1
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["topic_name"].add(
                frame[packet_identifier_len + i:packet_identifier_len + i + topic_name_result["str_len"] * 2 + 4])
            correct_flags_num = correct_flags_num + 1
            i = i + topic_name_result["str_len"] * 2 + 4

    if count == 0:  # 至少有一个topic_name
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num
        }
    return {
        "result": OK_ERROR,
        "correct_flags_num": correct_flags_num
    }


def verify_opt_head_and_payload_pingReq(frame, frame_flags, correct_flags_num):
    if len(frame) == 0:
        return {
            "result": OK_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_opt_head_and_payload_disconnect(frame, frame_flags, correct_flags_num):
    if len(frame) == 0:
        return {
            "result": OK_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num
        }


def verify_opt_head_and_payload(frame, frame_type, frame_flags, correct_flags_num):
    frame_type_func_map = {
        0x1: verify_opt_head_and_payload_connect,
        0x3: verify_opt_head_and_payload_publish,
        0x4: verify_opt_head_and_payload_puback,
        0x5: verify_opt_head_and_payload_pubrec,
        0x6: verify_opt_head_and_payload_pubrel,
        0x7: verify_opt_head_and_payload_pubcomp,
        0x8: verify_opt_head_and_payload_subscribe,
        0xa: verify_opt_head_and_payload_unsubscribe,
        0xc: verify_opt_head_and_payload_pingReq,
        0xe: verify_opt_head_and_payload_disconnect
    }
    if frame_type not in frame_type_func_map.keys():
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:

        return frame_type_func_map[frame_type](frame, frame_flags, correct_flags_num)
    # if frame_type_func_map[frame_type](frame,frame_flags)["result"] == OK_ERROR:
    #     return


def verify_frame(frame):
    global num_correct_flags
    global num_types_of_num_correct_flags
    correct_flags_num = 0
    full_correct = True
    frame_type_result = verify_frame_type(frame)
    if frame_type_result["result"] == OK_ERROR:
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["type"].add(frame[:1])
        correct_flags_num = correct_flags_num + 1
        frame = frame[1:]
    else:
        full_correct = False
        if frame_type_result["result"] == LEN_ERROR:
            return {
                "result": full_correct,
                "correct_flags_num": correct_flags_num
            }

    frame_flags_result = verify_frame_flags(frame, frame_type_result["frame_type"])
    if frame_flags_result["result"] == OK_ERROR:
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["flags"].add(frame[:1])
        correct_flags_num = correct_flags_num + 1
        frame = frame[1:]
    else:
        full_correct = False
        if frame_flags_result["result"] == LEN_ERROR:
            return {
                "result": full_correct,
                "correct_flags_num": correct_flags_num
            }

    len_result = verify_len(frame)
    if len_result["result"] == OK_ERROR:
        frame = frame[len_result["len_bytes_num"] * 2:]
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["len"].add(len_result["frame_len"])
        correct_flags_num = correct_flags_num + 1
    else:
        full_correct = False
        if len_result["result"] == LEN_ERROR:
            return {
                "result": full_correct,
                "correct_flags_num": correct_flags_num
            }
    verify_opt_head_and_payload_result = verify_opt_head_and_payload(frame, frame_type_result["frame_type"],
                                                                     frame_flags_result["frame_flags"],
                                                                     correct_flags_num)
    if not verify_opt_head_and_payload_result["result"] == OK_ERROR:
        return {
            "result": False,
            "correct_flags_num": verify_opt_head_and_payload_result["correct_flags_num"]
        }
    correct_flags_num = verify_opt_head_and_payload_result["correct_flags_num"]

    return {
        "result": full_correct,
        "correct_flags_num": correct_flags_num
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
                    else:
                        pass
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

    frame_file = r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\Fast_RoPEGAN\mqtt\fuzzing_data"
    filter_str = []
    filter_str.append("epoch")
    filter_str.append("best_model")
    verify_frame_folder(frame_file, filter_str)
    # frame_file = r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\all_accuracy_txt\Fast_RoPEGAN\mqtt\fuzzing_data\epoch20_generate_data.txt"
    # verify_frame_file(frame_file)
