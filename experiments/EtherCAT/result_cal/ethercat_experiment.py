"""
Ethercat实验分析
"""


class Ecat(object):
    def __init__(self, frame):
        self.result = self.parse_frame(frame)
        self.length = self.result.get("length")
        self.resvered = self.result.get("resvered")
        self.type = self.result.get("type")
        self.sub_message_list = self.result.get("sub_message_list")
        self.padding = self.result.get("padding")

    def parse_frame(self, frame):
        result = {}

        ## 解析ethercat header '0000 1100 0001 0000'
        if len(frame) >= 4:  # len
            tmp = format(int("0x" + frame[0:4], 16), "b")  # 取出头两个字节的二进制表示
            tmp = "0" * (16 - len(tmp)) + tmp
            tmp = tmp[8:] + tmp[:8]  # 大小端转化
            result["length"] = int(tmp[5:16], 2)
            result["resvered"] = int(tmp[4:5], 2)
            result["type"] = int(tmp[0:4], 2)
            frame = frame[4:]
        else:
            # result["length"] == None
            # result["resvered"] == None
            # result["type"] == None
            frame = ""
        ## 解析submessage   可能不止一个
        sub_message_list = []
        while len(frame) != 0:
            sub_message = {}
            # cmd
            if len(frame) >= 2:
                sub_message["cmd"] = int("0x" + frame[0:2], 16)
                frame = frame[2:]
            else:
                frame = ""
            # index
            if len(frame) >= 2:
                sub_message["index"] = int("0x" + frame[0:2], 16)
                frame = frame[2:]
            else:
                frame = ""
            # slave address
            if len(frame) >= 4:
                sub_message["slave_address"] = frame[0:4]
                frame = frame[4:]
            else:
                frame = ""
            # offset address
            if len(frame) >= 4:
                sub_message["offset_address"] = frame[0:4]
                frame = frame[4:]
            else:
                frame = ""
            # length reserved circulating more ecat.adp == 0x03eb && ecat.idx == 0xe7
            if len(frame) >= 4:
                tmp = format(int("0x" + frame[0:4], 16), "b")  # 取出头两个字节的二进制表示
                tmp = "0" * (16 - len(tmp)) + tmp
                tmp = tmp[8:] + tmp[:8]  # 大小端转化
                sub_message["length"] = int(tmp[5:16], 2)
                sub_message["resvered"] = int(tmp[2:5], 2)
                sub_message["type"] = int(tmp[1:2], 2)
                sub_message["more"] = int(tmp[0:1], 2)

                frame = frame[4:]
            else:
                frame = ""

            # irq
            if len(frame) >= 4:
                sub_message["irq"] = int("0x" + frame[0:4], 16)
                frame = frame[4:]
            else:
                frame = ""

            # data
            if len(frame) != 0 and len(frame) >= (sub_message["length"] * 2):
                sub_message["data"] = frame[:sub_message["length"] * 2]
                frame = frame[sub_message["length"] * 2:]
            else:
                frame = ""
            # wkc
            if len(frame) >= 4:
                sub_message["wkc"] = int("0x" + frame[2:4] + frame[0:2], 16)
                frame = frame[4:]
            else:
                frame = ""

            sub_message_list.append(sub_message)
            if not (sub_message.get("more") is not None and sub_message["more"] == 1):
                result["padding"] = frame
                break

        result["sub_message_list"] = sub_message_list
        return result


count = 0
unknown_attack_count = 0  # more后面没有子报文了 ，
packet_injection_count = 0
mitm_count = 0
wkc_attack_count = 0
slave_address_attack = 0  #


def detectEcat(frame_client_str, frame_py_server_str):
    global count
    global packet_injection_count
    global mitm_count
    global unknown_attack_count
    global wkc_attack_count
    global slave_address_attack  #
    slave_address_list = ["0000", "eb03", "ec03", "ea03", "e903"]
    frame_client = Ecat(frame_client_str)
    frame_py_server = Ecat(frame_py_server_str)
    for sub_mess_frame_client, sub_mess_frame_py_server in zip(frame_client.sub_message_list,
                                                               frame_py_server.sub_message_list):
        if sub_mess_frame_client.get("slave_address") not in slave_address_list or \
                sub_mess_frame_py_server.get("slave_address") not in slave_address_list:
            slave_address_attack = slave_address_attack + 1
        if sub_mess_frame_client.get("more") == 1 and frame_client.sub_message_list[-1] == sub_mess_frame_client:
            unknown_attack_count = unknown_attack_count + 1

        if sub_mess_frame_client.get("wkc") != sub_mess_frame_py_server.get("wkc"):
            if sub_mess_frame_client.get("length") == sub_mess_frame_py_server.get("length"):
                packet_injection_count = packet_injection_count + 1

            if sub_mess_frame_client.get("data") != sub_mess_frame_py_server.get("data"):
                mitm_count = mitm_count + 1

        else:
            if sub_mess_frame_client.get("data") != sub_mess_frame_py_server.get("data"):
                wkc_attack_count = wkc_attack_count + 1

    # if len(frame_client.sub_message_list) != len(frame_py_server.sub_message_list):
    #     count = count +1
    #     print("frame_client_str:",frame_client_str)
    #     print("frame_py_server_str:", frame_py_server_str)


#
#

with open(r"ethercat_attack_slave_2022_6_8.txt", "r") as client_f, open(
        r"ethercat_attack_slave_2022_6_8_Phy_server.txt", "r") as phy_server_f:
    client_f_list = client_f.readlines()
    phy_server_f_list = phy_server_f.readlines()
    for client_frame, phy_server_frame in zip(client_f_list, phy_server_f_list):
        detectEcat(client_frame, phy_server_frame)
        count = count + 1
        if count >= 63000:
            break
        # if len(client_frame) == len(phy_server_frame):
        #     count = count + 1

# print("count:", count)
print("unknown_attack_count:", unknown_attack_count)
print("packet_injection_count:", packet_injection_count)
print("mitm_count:", mitm_count)
print("slave_address_attack:", slave_address_attack)
print("wkc_attack_count:", wkc_attack_count)
if __name__ == "__main__":
    pass
    # print(int("0x"+"0003",16))
    # for frame in client_frame:
    #     ecat_message = Ecat(frame)
    #     print(ecat_message)
