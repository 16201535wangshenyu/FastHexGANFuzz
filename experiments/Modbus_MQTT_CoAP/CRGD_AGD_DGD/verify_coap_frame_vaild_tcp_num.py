import os
import struct
from tqdm import tqdm

# verify udp coap valid
CONFIRMABLE = 0
NON_CONFIRMABLE = 1
ACKNOWLEDGEMENT = 2
RESET = 3
#################### CODE ###########################
CODE_EMPTY = 0
CODE_GET = 1
CODE_POST = 2
CODE_PUT = 3
CODE_DELETE = 4
CODE_FETCH = 5
CODE_PATCH = 6
CODE_IPATCH = 7
#################### OPTION ###########################
COAP_OPTION_IF_MATCH = 1  #
COAP_OPTION_URI_HOST = 3  #
COAP_OPTION_ETAG = 4  #
COAP_OPTION_IF_NONE_MATCH = 5  #
COAP_OPTION_OBSERVE = 6  #
COAP_OPTION_URI_PORT = 7  #
COAP_OPTION_LOCATION_PATH = 8
COAP_OPTION_OSCORE = 9
COAP_OPTION_URI_PATH = 11  #
COAP_OPTION_CONTENT_FORMAT = 12  #
# COAP_OPTION_CONTENT_TYPE = 12
COAP_OPTION_MAXAGE = 14
COAP_OPTION_URI_QUERY = 15  #
COAP_OPTION_HOP_LIMIT = 16  #
COAP_OPTION_ACCEPT = 17  #
COAP_OPTION_LOCATION_QUERY = 20
COAP_OPTION_BLOCK2 = 23  #
COAP_OPTION_BLOCK1 = 27  #
COAP_OPTION_SIZE2 = 28  #
COAP_OPTION_PROXY_URI = 35  #
COAP_OPTION_PROXY_SCHEME = 39  #
COAP_OPTION_SIZE1 = 60  #
COAP_OPTION_NORESPONSE = 258  #

####################MODIEA_TYPE##########################
COAP_MEDIATYPE_TEXT_PLAIN = 0  # /* text/plain (UTF-8) */
COAP_MEDIATYPE_APPLICATION_LINK_FORMAT = 40  # /* application/link-format */
COAP_MEDIATYPE_APPLICATION_XML = 41  # /* application/xml */
COAP_MEDIATYPE_APPLICATION_OCTET_STREAM = 42  # /* application/octet-stream */
COAP_MEDIATYPE_APPLICATION_RDF_XML = 43  # /* application/rdf+xml */
COAP_MEDIATYPE_APPLICATION_EXI = 47  # /* application/exi  */
COAP_MEDIATYPE_APPLICATION_JSON = 50  # /* application/json  */
COAP_MEDIATYPE_APPLICATION_CBOR = 60  # /* application/cbor  */
COAP_MEDIATYPE_APPLICATION_CWT = 61  # /* application/cwt, RFC 8392  */

COAP_MEDIATYPE_APPLICATION_COSE_SIGN = 98  # /* application/cose; cose-type="cose-sign"     */
COAP_MEDIATYPE_APPLICATION_COSE_SIGN1 = 18  # /* application/cose; cose-type="cose-sign1"    */
COAP_MEDIATYPE_APPLICATION_COSE_ENCRYPT = 96  # /* application/cose; cose-type="cose-encrypt"  */
COAP_MEDIATYPE_APPLICATION_COSE_ENCRYPT0 = 16  # /* application/cose; cose-type="cose-encrypt0" */
COAP_MEDIATYPE_APPLICATION_COSE_MAC = 97  # /* application/cose; cose-type="cose-mac"      */
COAP_MEDIATYPE_APPLICATION_COSE_MAC0 = 17  # /* application/cose; cose-type="cose-mac0"     */
COAP_MEDIATYPE_APPLICATION_COSE_KEY = 101  # /* application/cose-key  */
COAP_MEDIATYPE_APPLICATION_COSE_KEY_SET = 102  # /* application/cose-key-set  */

COAP_MEDIATYPE_APPLICATION_SENML_JSON = 110  # /* application/senml+json  */
COAP_MEDIATYPE_APPLICATION_SENSML_JSON = 111  # /* application/sensml+json */
COAP_MEDIATYPE_APPLICATION_SENML_CBOR = 112  # /* application/senml+cbor  */
COAP_MEDIATYPE_APPLICATION_SENSML_CBOR = 113  # /* application/sensml+cbor */
COAP_MEDIATYPE_APPLICATION_SENML_EXI = 114  # /* application/senml-exi   */
COAP_MEDIATYPE_APPLICATION_SENSML_EXI = 115  # /* application/sensml-exi  */
COAP_MEDIATYPE_APPLICATION_SENML_XML = 310  # /* application/senml+xml   */
COAP_MEDIATYPE_APPLICATION_SENSML_XML = 311  # /* application/sensml+xml  */
COAP_MEDIATYPE_APPLICATION_DOTS_CBOR = 271  # /* application/dots+cbor */
COAP_MEDIATYPE_ANY = 0xff
###################### SIGNAL CODE ###########################
COAP_SIGNALING_CSM = 225  # 111 00001
COAP_SIGNALING_PING = 226
COAP_SIGNALING_PONG = 227
COAP_SIGNALING_RELEASE = 228
COAP_SIGNALING_ABORT = 229
###################### SIGNAL OPTION ##########################
COAP_SIGNAL_OPTION_MAX_MSG_SIZE = 2
COAP_SIGNAL_OPTION_BLOCKWISE_TRANSFER = 4
COAP_SIGNAL_OPTION_CUSTODY = 2
COAP_SIGNAL_OPTION_ALT_ADDR = 2
COAP_SIGNAL_OPTION_HOLD_OFF = 4
COAP_SIGNAL_OPTION_BAD_CSM = 2
###ERROR_CODE######
OK_ERROR = 0
LEN_ERROR = -1
FIELD_ERROR = -2

num_correct_flags = [0 for r in range(9000)]
num_types_of_num_correct_flags = {
    "Len": set(),  # len与extend_len当作一个字段，len
    "TKL": set(),
    "code": set(),
    "token": set(),
    "option_delta": set(),
    "option_len": set(),
    "opt_IF_MATCH_value": set(),
    "opt_URI_HOST_value": set(),
    "opt_ETAG_value": set(),
    "opt_IF_NONE_MATCH_value": set(),
    "opt_OBSERVE_value": set(),
    "opt_URI_PORT_value": set(),
    "opt_URI_PATH_value": set(),
    "opt_CONTENT_FORMAT_value": set(),
    "opt_URI_QUERY_value": set(),
    "opt_HOP_LIMIT_value": set(),
    "opt_ACCEPT_value": set(),
    "opt_BLOCK2_value": set(),
    "opt_BLOCK1_value": set(),
    "opt_SIZE2_value": set(),
    "opt_PROXY_URI_value": set(),
    "opt_PROXY_SCHEME_value": set(),
    "opt_SIZE1_value": set(),
    "opt_NORESPONSE_value": set(),
    "opt_SIGNAL_MAX_MSG_SIZE": set(),
    "opt_SIGNAL_BLOCKWISE_TRANSFER": set(),
    "opt_SIGNAL_CUSTODY": set(),
    "opt_SIGNAL_ALT_ADDR": set(),
    "opt_SIGNAL_HOLD_OFF": set(),
    "opt_SIGNAL_BAD_CSM": set(),
    "payload": set()

}


def string_to_bytes(string):
    str1 = ''
    str2 = b''
    while string:
        str1 = string[0:2]
        s = int(str1, 16)
        str2 += struct.pack('B', s)
        string = string[2:]
    return str2


def verify_Len(frame):
    num_bytes_of_extend_len = 0
    if len(frame) >= 1:

        len_flag = int("0x" + frame[0], 16)

        # parse len
        if len_flag <= 12:
            len_flag = len_flag

        elif len_flag > 12 and len_flag <= 268:  # 代表有1个字节的len_extend
            if len(frame) >= 4:  # 代表拥有extend_len字段
                extend_len = int("0x" + frame[2:4], 16)
                len_flag = extend_len + 13
                num_bytes_of_extend_len = 1
            else:  # 不拥有extend_len字段
                return {
                    "result": LEN_ERROR
                }
        elif len_flag > 268 and len_flag <= 65804:  # 代表有2个字节的len_extend
            if len(frame) >= 6:  # 代表拥有extend_len字段
                extend_len = int("0x" + frame[2:6], 16)
                len_flag = extend_len + 269
                num_bytes_of_extend_len = 2
            else:  # 不拥有extend_len字段
                return {
                    "result": LEN_ERROR
                }
        elif len_flag > 65804:  # 代表有4个字节的len_extend
            if len(frame) >= 10:  # 代表拥有extend_len字段
                extend_len = int("0x" + frame[2:10], 16)
                len_flag = extend_len + 65805
                num_bytes_of_extend_len = 4
            else:  # 不拥有extend_len字段
                return {
                    "result": LEN_ERROR
                }
        # 现在得到了len_flag，对比一下，看一下len字段的值是否正确
        if len(frame) >= 2:  # 确保拥有Len以及TKL两个字段
            if (len(frame) // 2 - 1 - num_bytes_of_extend_len - 1 - int("0x" + frame[1],
                                                                        16)) == len_flag:  # frame 有len字段这么长
                return {
                    "result": OK_ERROR,
                    "value": num_bytes_of_extend_len,
                    "len_flag": len_flag
                }
            else:  # 如果没有len字段那么长
                return {
                    "result": FIELD_ERROR,
                    "value": num_bytes_of_extend_len
                }

        else:
            return {
                "result": LEN_ERROR
            }
    else:
        return {
            "result": LEN_ERROR
        }


def verify_TKL(frame):
    if len(frame) >= 1:
        token_len = int("0x" + frame[0], 16)

        if token_len in [0, 1, 2, 4]:
            return {
                "result": OK_ERROR,
                "value": token_len
            }
        else:
            return {
                "result": FIELD_ERROR,
                "value": None
            }
    else:
        return {
            "result": LEN_ERROR,
            "value": None
        }


def verify_Code(frame):
    if len(frame) >= 2:
        code_tye = int("0x" + frame[0] + frame[1], 16)

        if code_tye in [CODE_EMPTY, CODE_POST, CODE_GET, CODE_DELETE, CODE_PATCH, CODE_PUT, CODE_IPATCH, CODE_FETCH,
                        COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:
            return {
                "result": OK_ERROR,
                "value": code_tye
            }
        else:
            return {
                "result": FIELD_ERROR,
                "value": None
            }
    else:
        return {
            "result": LEN_ERROR,
            "value": None
        }


def verify_messageID(frame):
    if len(frame) >= 4:
        return {
            "result": OK_ERROR,
            "value": None
        }
    else:
        return {
            "result": LEN_ERROR,
            "value": None
        }


def verify_token(frame, token_len):
    if len(frame) >= token_len * 2:
        return {
            "result": OK_ERROR,
            "value": None
        }
    else:
        return {
            "result": LEN_ERROR,
            "value": None
        }


def parse_opt_delta(frame):
    if len(frame) >= 1:
        delta = int("0x" + frame[0], 16)
        if delta < 13:
            return {
                "result": OK_ERROR,
                "value": {
                    "delta": delta,
                    "extend_len": 0
                }
            }
        elif delta == 13:  #
            if len(frame) >= 4:
                delta_extend = int("0x" + frame[2] + frame[3], 16)
                delta = delta_extend + 13
                return {
                    "result": OK_ERROR,
                    "value": {
                        "delta": delta,
                        "extend_len": 1
                    }
                }
            else:
                return {
                    "result": LEN_ERROR,
                    "value": None
                }
        elif delta == 14:
            if len(frame) >= 6:
                delta_extend = int("0x" + frame[2] + frame[3] + frame[4] + frame[5], 16)
                delta = delta_extend + 269
                return {
                    "result": OK_ERROR,
                    "value": {
                        "delta": delta,
                        "extend_len": 2
                    }
                }
            else:
                return {
                    "result": LEN_ERROR,
                    "value": None
                }
        else:
            return {
                "result": FIELD_ERROR,
                "value": None
            }
    else:
        return {
            "result": LEN_ERROR,
            "value": None
        }


def parse_opt_length(frame, delta):
    delta_extend_bytes = delta["extend_len"]
    # if delta >= 13 and delta <=268:
    #     delta_extend_bytes = 1
    # elif delta > 268:
    #     delta_extend_bytes = 2

    if len(frame) >= 2:
        opt_len = int("0x" + frame[1], 16)
        if opt_len < 13:
            opt_total_len = 2 + delta["extend_len"] * 2 + 0 * 2 + opt_len * 2
            if len(frame) >= opt_total_len:
                return {
                    "result": OK_ERROR,
                    "value": {
                        "opt_len": opt_len,
                        'extend_len': 0
                    }

                }
            else:
                return {
                    "result": LEN_ERROR,
                    "value": None
                }

        elif opt_len == 13:  #
            if len(frame) >= 2 + delta_extend_bytes * 2 + 2:
                opt_len_extend = int("0x" + frame[2 + delta_extend_bytes * 2] + frame[2 + delta_extend_bytes * 2 + 1],
                                     16)
                opt_len = opt_len_extend + 13

                opt_total_len = 2 + delta["extend_len"] * 2 + 1 * 2 + opt_len * 2
                if len(frame) >= opt_total_len:
                    return {
                        "result": OK_ERROR,
                        "value": {
                            "opt_len": opt_len,
                            'extend_len': 1
                        }
                    }
                else:
                    return {
                        "result": LEN_ERROR,
                        "value": None
                    }
            else:
                return {
                    "result": LEN_ERROR,
                    "value": None
                }
        elif opt_len == 14:
            if len(frame) >= 6 + delta_extend_bytes * 2:
                opt_len_extend = int(
                    "0x" + frame[2 + delta_extend_bytes * 2] + frame[2 + delta_extend_bytes * 2 + 1] + frame[
                        2 + delta_extend_bytes * 2 + 2] + frame[2 + delta_extend_bytes * 2 + 3], 16)
                opt_len = opt_len_extend + 269
                opt_total_len = 2 + delta["extend_len"] * 2 + 2 * 2 + opt_len * 2
                if len(frame) >= opt_total_len:
                    return {
                        "result": OK_ERROR,
                        "value": {
                            "opt_len": opt_len,
                            'extend_len': 2
                        }
                    }
                else:
                    return {
                        "result": LEN_ERROR,
                        "value": None
                    }
            else:
                return {
                    "result": LEN_ERROR,
                    "value": None
                }
        else:
            return {
                "result": LEN_ERROR,
                "value": None
            }
    else:
        return {
            "result": LEN_ERROR,
            "value": None
        }


"""

frame
code_type
delta['value']
"""


def verify_opt_ifmatch(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_len = parse_opt_length(frame, delta)
    opt_total_len = 0
    if opt_len["result"] == OK_ERROR:  # 错误比较致命，全部都是属于LEN_ERROR
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if opt_len['value']['opt_len'] > 3:
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num,
                "opt_total_len": opt_total_len,

            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2:opt_total_len],
            "correct_flags_num": correct_flags_num
        }


    else:
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num,
            "opt_total_len": opt_total_len
        }


"""
loosen restrictions ,don't require: host must 127.0.0.1 ,but require len is len(127.0.0.1) = 9
"""


def verify_opt_uri_host(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if opt_len['value']['opt_len'] != 9:
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num,
                "opt_total_len": opt_total_len
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2:opt_total_len],
            "correct_flags_num": correct_flags_num,
        }
    else:
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num,
            "opt_total_len": opt_total_len,
        }


def verify_opt_etag(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:
        opt_len = parse_opt_length(frame, delta)

        if not 1 <= opt_len['value']['opt_len'] <= 3:  #
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num,
                "opt_total_len": opt_total_len
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2:opt_total_len],
            "correct_flags_num": correct_flags_num
        }


    else:
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num,
            "opt_total_len": opt_total_len
        }


def verify_opt_if_none_match(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type == CODE_POST:

        if not opt_len['value']['opt_len'] == 0:  #
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num,
                "opt_total_len": opt_total_len
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len],
            "correct_flags_num": correct_flags_num
        }


    else:
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num,
            "opt_total_len": opt_total_len
        }


def verify_opt_observe(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        try:
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
            correct_flags_num = correct_flags_num + 1
        except Exception as e:
            print(e)
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }

    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if not opt_len['value']['opt_len'] in [0, 1]:  #
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num,
                "opt_total_len": opt_total_len
            }

        if opt_len['value']['opt_len'] == 1:  # observe_value must be 0|1
            opt_value = int(
                "0x" + frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2] + frame[
                    2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 1], 16)
            if opt_value in [0, 1]:
                return {
                    "result": OK_ERROR,
                    "opt_total_len": opt_total_len,
                    "value": frame[
                             2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len],
                    "correct_flags_num": correct_flags_num
                }
            else:
                return {
                    "result": FIELD_ERROR,
                    "correct_flags_num": correct_flags_num,
                    "opt_total_len": opt_total_len

                }
        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": ""
        }


    else:
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num,
            "opt_total_len": opt_total_len,
        }


def verify_opt_uri_port(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }

    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if not opt_len['value']['opt_len'] == 2:  # uri_port must be 5683
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num,
                "opt_total_len": opt_total_len
            }

        opt_value = int("0x" + frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2] +
                        frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 1] +
                        frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 2] +
                        frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 3]
                        , 16)
        if opt_value == 5683:
            return {
                "result": OK_ERROR,
                "opt_total_len": opt_total_len,
                "value": frame[
                         2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len],
                "correct_flags_num": correct_flags_num
            }
        else:
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num,
                "opt_total_len": opt_total_len
            }



    else:
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num,
            "opt_total_len": opt_total_len
        }


"""
验证uri_path是否正确
uri_path：验证 opt_len 在 0-255字节，且报文足够长即可
"""


def verify_opt_uri_path(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if opt_len['value']['opt_len'] > 255:  #
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num,
                "opt_total_len": opt_total_len

            }

        # uri_path_value = string_to_bytes(frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 : opt_total_len]).decode("ascii")

        return {
            "result": OK_ERROR,
            "correct_flags_num": correct_flags_num,
            "opt_total_len": opt_total_len,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }


    else:
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num,
            "opt_total_len": opt_total_len
        }


"""
验证uri_query字段是否正确
uri_query：验证 opt_len 在 1-255字节，且报文足够长即可
"""


def verify_opt_uri_query(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if not 1 <= opt_len['value']['opt_len'] <= 255:  # uri_port must be 5683
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num,
                "opt_total_len": opt_total_len
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }


    else:
        return {
            "result": FIELD_ERROR,
            "correct_flags_num": correct_flags_num,
            "opt_total_len": opt_total_len
        }


"""
hot limit
hot limit：opt_len 1个字节
"""


def verify_opt_hop_limit(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if not 1 == opt_len['value']['opt_len']:  # uri_port must be 5683
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len],
            "correct_flags_num": correct_flags_num
        }
    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


"""
COAP_OPTION_BLOCK2
code empty,为None，
code 其它，有或者没有 opt_len随机0-3字节 
"""


def verify_opt_block2(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if not 0 <= opt_len['value']['opt_len'] <= 3:  # uri_port must be 5683
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }


    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


"""
COAP_OPTION_BLOCK1
code empty,为None，
code 其它，有或者没有 opt_len 随机0-3字节 
"""


def verify_opt_block1(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if not 0 <= opt_len['value']['opt_len'] <= 3:  # uri_port must be 5683
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }


    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


"""
code empty,为None，
code 其它，有或者没有 随机0-4字节 
"""


def verify_opt_size2(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if not 0 <= opt_len['value']['opt_len'] <= 4:  # uri_port must be 5683
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]

        }


    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


"""
code empty,为None，
code 其它，有或者没有 随机1-1034字节 
"""


def verify_opt_proxy_uri(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:
        opt_len = parse_opt_length(frame, delta)

        if not 1 <= opt_len['value']['opt_len'] <= 1034:  #
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }


    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


"""
code empty,为None，
code 其它，有或者没有 随机1-255字节 
"""


def verify_opt_proxy_scheme(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if not 1 <= opt_len['value']['opt_len'] <= 255:  # uri_port must be 5683
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }




    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


"""
code empty,为None，
code 其它，有或者没有 随机0-4字节 
"""


def verify_opt_size1(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if not 0 <= opt_len['value']['opt_len'] <= 4:  # uri_port must be 5683
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num,

            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }


    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


def verify_opt_noresponse(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if not opt_len['value']['opt_len'] in [0, 1]:  # uri_port must be 5683
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

        if opt_len['value']['opt_len'] == 1:
            opt_value = int(
                "0x" + frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2] + frame[
                    2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 1], 16)
            if opt_value in [2, 8, 16]:
                return {
                    "result": OK_ERROR,
                    "opt_total_len": opt_total_len,
                    "correct_flags_num": correct_flags_num,
                    "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
                }
            else:
                return {
                    "result": FIELD_ERROR,
                    "opt_total_len": opt_total_len,
                    "correct_flags_num": correct_flags_num
                }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }


    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


all_media_type = [
    COAP_MEDIATYPE_TEXT_PLAIN,  # 当为0字节长度时，默认为COAP_MEDIATYPE_TEXT_PLAIN
    COAP_MEDIATYPE_APPLICATION_LINK_FORMAT,
    COAP_MEDIATYPE_APPLICATION_XML,
    COAP_MEDIATYPE_APPLICATION_OCTET_STREAM,
    COAP_MEDIATYPE_APPLICATION_RDF_XML,
    COAP_MEDIATYPE_APPLICATION_EXI,
    COAP_MEDIATYPE_APPLICATION_JSON,
    COAP_MEDIATYPE_APPLICATION_CBOR,
    COAP_MEDIATYPE_APPLICATION_CWT,
    COAP_MEDIATYPE_APPLICATION_COSE_SIGN,
    COAP_MEDIATYPE_APPLICATION_COSE_SIGN1,
    COAP_MEDIATYPE_APPLICATION_COSE_ENCRYPT,
    COAP_MEDIATYPE_APPLICATION_COSE_ENCRYPT0,
    COAP_MEDIATYPE_APPLICATION_COSE_MAC,
    COAP_MEDIATYPE_APPLICATION_COSE_MAC0,
    COAP_MEDIATYPE_APPLICATION_COSE_KEY,
    COAP_MEDIATYPE_APPLICATION_COSE_KEY_SET,
    COAP_MEDIATYPE_APPLICATION_SENML_JSON,
    COAP_MEDIATYPE_APPLICATION_SENSML_JSON,
    COAP_MEDIATYPE_APPLICATION_SENML_CBOR,
    COAP_MEDIATYPE_APPLICATION_SENSML_CBOR,
    COAP_MEDIATYPE_APPLICATION_SENML_EXI,
    COAP_MEDIATYPE_APPLICATION_SENSML_EXI,
    COAP_MEDIATYPE_APPLICATION_SENML_XML,
    COAP_MEDIATYPE_APPLICATION_SENSML_XML,
    COAP_MEDIATYPE_APPLICATION_DOTS_CBOR,
    COAP_MEDIATYPE_ANY,
]
"""

"""


def verify_opt_content_format(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    global all_media_type
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, CODE_GET, CODE_DELETE, COAP_SIGNALING_CSM, COAP_SIGNALING_PING,
                         COAP_SIGNALING_PONG, COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if not opt_len['value']['opt_len'] in [0, 1, 2]:
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

        opt_value = 0
        if opt_len['value']['opt_len'] == 1:
            opt_value = int(
                "0x" + frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2] + frame[
                    2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 1], 16)
        elif opt_len['value']['opt_len'] == 2:  # opt_len ==2
            opt_value = int("0x" + frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2] +
                            frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 1] +
                            frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 2] +
                            frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 3]
                            , 16)
        else:
            return {
                "result": OK_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num,
                "value": ""
            }
        if opt_value in all_media_type:
            return {
                "result": OK_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num,
                "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
            }
        else:
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


"""
"""


def verify_opt_accept(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type not in [CODE_EMPTY, COAP_SIGNALING_CSM, COAP_SIGNALING_PING, COAP_SIGNALING_PONG,
                         COAP_SIGNALING_RELEASE, COAP_SIGNALING_ABORT]:

        if not opt_len['value']['opt_len'] in [0, 1, 2]:  # uri_port must be 5683
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

        opt_value = 0
        if opt_len['value']['opt_len'] == 1:
            opt_value = int(
                "0x" + frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2] + frame[
                    2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 1], 16)
        elif opt_len['value']['opt_len'] == 2:  # opt_len ==2
            opt_value = int("0x" + frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2] +
                            frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 1] +
                            frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 2] +
                            frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + 3]
                            , 16)
        else:
            return {
                "result": OK_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num,
                "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
            }
        if opt_value in all_media_type:
            return {
                "result": OK_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num,
                "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]

            }
        else:
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }
    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


def verify_opt_max_msg_size(frame, code_type, delta, correct_flags_num):  # 条件放松，COAP_SIGNALING_CSM 可能具有max_msg_size
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type == COAP_SIGNALING_CSM:  # COAP_SIGNALING_CSM "必须" 具有该字段

        if not 0 <= opt_len['value']['opt_len'] <= 4:  #
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }
    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


#
def verify_opt_blockwise_transfer(frame, code_type, delta,
                                  correct_flags_num):  # 条件放松，COAP_SIGNALING_CSM 可能具有blockwise_transfer
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type == COAP_SIGNALING_CSM:

        if not opt_len['value']['opt_len'] == 0:  #
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }
        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }


    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


def verify_opt_custody(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type == COAP_SIGNALING_PING:

        if not opt_len['value']['opt_len'] == 0:  #
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]

        }


    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


def verify_opt_alt_addr(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type == COAP_SIGNALING_RELEASE:

        if not 1 <= opt_len['value']['opt_len'] <= 255:  # uri_port must be
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }
        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }

    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num

        }


def verify_opt_hold_off(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type == COAP_SIGNALING_RELEASE:  # COAP_SIGNALING_CSM "必须" 具有该字段

        if not 0 <= opt_len['value']['opt_len'] <= 3:  #
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }
        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }


    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


def verify_opt_bad_csm(frame, code_type, delta, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    opt_total_len = 0
    opt_len = parse_opt_length(frame, delta)
    if opt_len["result"] == OK_ERROR:
        opt_total_len = 2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2 + opt_len['value'][
            'opt_len'] * 2
        num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        num_types_of_num_correct_flags["option_len"].add(opt_len['value']['opt_len'])
        correct_flags_num = correct_flags_num + 1
    else:
        return {
            "result": opt_len["result"],
            "correct_flags_num": correct_flags_num
        }
    if code_type == COAP_SIGNALING_ABORT:  # COAP_SIGNALING_CSM "必须" 具有该字段

        if not 0 <= opt_len['value']['opt_len'] <= 2:  #
            return {
                "result": FIELD_ERROR,
                "opt_total_len": opt_total_len,
                "correct_flags_num": correct_flags_num
            }

        return {
            "result": OK_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num,
            "value": frame[2 + delta["extend_len"] * 2 + opt_len['value']['extend_len'] * 2: opt_total_len]
        }


    else:
        return {
            "result": FIELD_ERROR,
            "opt_total_len": opt_total_len,
            "correct_flags_num": correct_flags_num
        }


def verify_option(frame, code_type, correct_flags_num):
    global num_correct_flags
    global num_types_of_num_correct_flags
    full_correct = OK_ERROR
    option_func_dict = {
        COAP_OPTION_IF_MATCH: [verify_opt_ifmatch, "opt_IF_MATCH_value"],
        COAP_OPTION_URI_HOST: [verify_opt_uri_host, "opt_URI_HOST_value"],
        COAP_OPTION_ETAG: [verify_opt_etag, "opt_ETAG_value"],
        COAP_OPTION_IF_NONE_MATCH: [verify_opt_if_none_match, "opt_IF_NONE_MATCH_value"],
        COAP_OPTION_OBSERVE: [verify_opt_observe, "opt_OBSERVE_value"],
        COAP_OPTION_URI_PORT: [verify_opt_uri_port, "opt_URI_PORT_value"],
        COAP_OPTION_URI_PATH: [verify_opt_uri_path, "opt_URI_PATH_value"],
        COAP_OPTION_CONTENT_FORMAT: [verify_opt_content_format, "opt_CONTENT_FORMAT_value"],
        COAP_OPTION_URI_QUERY: [verify_opt_uri_query, "opt_URI_QUERY_value"],
        COAP_OPTION_HOP_LIMIT: [verify_opt_hop_limit, "opt_HOP_LIMIT_value"],
        COAP_OPTION_ACCEPT: [verify_opt_accept, "opt_ACCEPT_value"],
        COAP_OPTION_BLOCK2: [verify_opt_block2, "opt_BLOCK2_value"],
        COAP_OPTION_BLOCK1: [verify_opt_block1, "opt_BLOCK1_value"],
        COAP_OPTION_SIZE2: [verify_opt_size2, "opt_SIZE2_value"],
        COAP_OPTION_PROXY_URI: [verify_opt_proxy_uri, "opt_PROXY_URI_value"],
        COAP_OPTION_PROXY_SCHEME: [verify_opt_proxy_scheme, "opt_PROXY_SCHEME_value"],
        COAP_OPTION_SIZE1: [verify_opt_size1, "opt_SIZE1_value"],
        COAP_OPTION_NORESPONSE: [verify_opt_noresponse, "opt_NORESPONSE_value"]
    }
    csm_func_dict = {
        COAP_SIGNAL_OPTION_MAX_MSG_SIZE: [verify_opt_max_msg_size, "opt_SIGNAL_MAX_MSG_SIZE"],
        COAP_SIGNAL_OPTION_BLOCKWISE_TRANSFER: [verify_opt_blockwise_transfer, "opt_SIGNAL_BLOCKWISE_TRANSFER"],
    }
    ping_func_dict = {
        COAP_SIGNAL_OPTION_CUSTODY: [verify_opt_custody, "opt_SIGNAL_CUSTODY"]

    }
    release_func_dict = {
        COAP_SIGNAL_OPTION_ALT_ADDR: [verify_opt_alt_addr, "opt_SIGNAL_ALT_ADDR"],
        COAP_SIGNAL_OPTION_HOLD_OFF: [verify_opt_hold_off, "opt_SIGNAL_HOLD_OFF"]
    }
    abort_func_dict = {
        COAP_SIGNAL_OPTION_BAD_CSM: [verify_opt_bad_csm, "opt_SIGNAL_BAD_CSM"]
    }
    has_opt_content_format = False
    old_opt_code = 0
    # if frame == '6665447401006500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000040e12f80010040e140000000':
    #     print(frame)
    while len(frame) >= 2 and frame[0:2] != "ff":

        delta = parse_opt_delta(frame)
        if delta['result'] != OK_ERROR:
            return {
                "result": delta['result'],
                "correct_flags_num": correct_flags_num
            }
        # num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
        # correct_flags_num = correct_flags_num + 1
        # 'd004b4636f6170d308e2d960'
        delta = delta['value']
        func_dict = None
        if code_type == COAP_SIGNALING_CSM:
            func_dict = csm_func_dict
        elif code_type == COAP_SIGNALING_PING:
            func_dict = ping_func_dict
        elif code_type == COAP_SIGNALING_RELEASE:
            func_dict = release_func_dict
        elif code_type == COAP_SIGNALING_ABORT:
            func_dict = abort_func_dict
        else:
            func_dict = option_func_dict
        if delta['delta'] + old_opt_code in func_dict.keys():
            if delta['delta'] + old_opt_code == COAP_OPTION_CONTENT_FORMAT:
                has_opt_content_format = True
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1  # option delta字段正确
            num_types_of_num_correct_flags["option_delta"].add(delta['delta'])
            correct_flags_num = correct_flags_num + 1
            result = func_dict[delta['delta'] + old_opt_code][0](frame, code_type, delta, correct_flags_num)
            correct_flags_num = result["correct_flags_num"]
            if result["result"] == OK_ERROR:
                frame = frame[result["opt_total_len"]:]
                num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
                num_types_of_num_correct_flags[func_dict[delta['delta'] + old_opt_code][1]].add(result["value"])
                correct_flags_num = correct_flags_num + 1
            else:
                if result["result"] == LEN_ERROR:
                    return {
                        "result": result["result"],
                        "correct_flags_num": correct_flags_num
                    }
                else:
                    frame = frame[result["opt_total_len"]:]
                    full_correct = FIELD_ERROR
        else:
            return {
                "result": FIELD_ERROR,
                "correct_flags_num": correct_flags_num
            }

        old_opt_code = delta['delta'] + old_opt_code
    if len(frame) == 1:
        return {
            "result": LEN_ERROR,
            "correct_flags_num": correct_flags_num
        }
    else:  # frame == 0
        return {
            "result": OK_ERROR | full_correct,
            "has_opt_content_format": has_opt_content_format,
            "frame": frame,
            "correct_flags_num": correct_flags_num
        }


def verify_payload(frame, has_opt_content_format):
    if len(frame) == 0:
        return {
            "result": OK_ERROR
        }
    if not has_opt_content_format:
        return {
            "result": FIELD_ERROR
        }
    else:
        return {
            "result": OK_ERROR
        }


def verify_frame(frame):
    correct_flags_num = 0
    full_correct = True

    global num_correct_flags
    if frame is not None and frame != "":
        token_len = 0
        code_type = 0

        verify_Len_result = verify_Len(frame)
        if verify_Len_result["result"] == OK_ERROR:
            frame = frame[1:]
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["Len"].add(verify_Len_result["len_flag"])
            correct_flags_num = correct_flags_num + 1
        else:
            full_correct = False
            if verify_Len_result["result"] == LEN_ERROR:
                return {
                    "result": full_correct,
                    "correct_flags_num": correct_flags_num
                }

        verify_TKL_result = verify_TKL(frame)

        if verify_TKL_result["result"] == OK_ERROR:
            frame = frame[1:]
            token_len = verify_TKL_result["value"]
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["TKL"].add(token_len)
            correct_flags_num = correct_flags_num + 1
        else:
            full_correct = False
            if verify_TKL_result["result"] == LEN_ERROR:
                return {
                    "result": full_correct,
                    "correct_flags_num": correct_flags_num
                }

        frame = frame[verify_Len_result["value"] * 2:]  # 跳过extend_len字段

        code_result = verify_Code(frame)
        if code_result["result"] == OK_ERROR:
            frame = frame[2:]
            code_type = code_result["value"]
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["code"].add(code_type)
            correct_flags_num = correct_flags_num + 1
        else:
            full_correct = False
            if code_result["result"] == LEN_ERROR:
                return {
                    "result": full_correct,
                    "correct_flags_num": correct_flags_num
                }

        if verify_token(frame, token_len)["result"] == OK_ERROR:
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["token"].add(frame[0:token_len * 2])
            correct_flags_num = correct_flags_num + 1
            frame = frame[token_len * 2:]
        else:
            full_correct = False
            return {
                "result": full_correct,
                "correct_flags_num": correct_flags_num
            }

        option_result = verify_option(frame, code_type, correct_flags_num)
        if option_result["result"] != OK_ERROR:
            full_correct = False
            return {
                "result": full_correct,
                "correct_flags_num": option_result["correct_flags_num"]
            }

        frame = option_result["frame"]
        correct_flags_num = option_result["correct_flags_num"]
        if verify_payload(frame, option_result["has_opt_content_format"])["result"] == OK_ERROR:
            num_correct_flags[correct_flags_num] = num_correct_flags[correct_flags_num] + 1
            num_types_of_num_correct_flags["payload"].add(frame)
            correct_flags_num = correct_flags_num + 1
            return {
                "result": full_correct,
                "correct_flags_num": correct_flags_num
            }
        else:
            return {
                "result": False,
                "correct_flags_num": correct_flags_num
            }
    else:
        return {
            "result": False,
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

    num_correct_flags = [0 for r in range(9000)]  # 清空num_correct_flags
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
    # folder_path = r"G:\研究生\研究生\科研\小论文\FCS-21046_Proof_hi\代码2\WGAN\output\coap\train1"
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

    frame_file = r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\frame_folder\coap"
    # verify_frame_file(frame_file)
    filter_str = []
    filter_str.append("")
    filter_str.append("")
    verify_frame_folder(frame_file, filter_str)

    # folder_path = r"G:\研究生\研究生\科研\小论文\FCS-21046_Proof_hi\代码2\WGAN\output\coap\lstm"
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
