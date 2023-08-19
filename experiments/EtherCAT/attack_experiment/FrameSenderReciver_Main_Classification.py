# encoding: utf-8
import socket
import struct
import time
import os
from tqdm import tqdm
from ethernet import EthernetSocket
""" 
#####   执行目录为 ..\SAGAN_MQTT; 
报文的功能全部为发布消息的报文
"""

def dataSwitch(data):
    str1 = ''
    str2 = b''#############################################################################################################
    while data:
        str1 = data[0:2]
        s = int(str1,16)
        str2 += struct.pack('!B',s)
        data = data[2:]
    return str2



def ECAT_collect2server():
    global ethercat_sock
    
    ethercat_sock = EthernetSocket()
    req_val = "0c1000df000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    # switch hex to bit for sending it to the
    packet = dataSwitch(req_val.strip('\n'))
    

    ethercat_sock.send(packet, EthernetSocket.ETH_CAT)
    
    time.sleep(0.1)

    data = ethercat_sock.recv(None, EthernetSocket.ETH_CAT)
    print("recv:", data)
    
    # req_val = "bdf205de9cd4634900000000a002ffd7fe3000000204ffd70402080a63bd1c4a0000000001030307"
    # req_string = dataSwitch(req_val.strip('\n'))  # switch hex to bit for sending it to the simulations
    # s.send(req_string)
    # time.sleep(0.1)
    # req_ack = s.recv(BUFFER_SIZE).hex()  # 获取连接请求回应数据包



if __name__ == '__main__':
    train_data_path = r'/home/wangshenyu/application/paper_fuzzing/attack_server/modbus_attack_server/Data/output/ethercat/test20220420_1_ethercat_10_0F_17'
    # train_data_path = 'Data/output/fatal_error'
    # train_data_path = 'Data/output'
    seek_number = 10240

    global sum_count
    global BUFFER_SIZE

    global ethercat_sock
    BUFFER_SIZE = 10000

    files_dict = {}
    if os.path.isdir(train_data_path):
        print("it's a directory")
        for file_name in os.listdir(train_data_path):
            file_path = train_data_path + "/" + file_name
            files_dict[file_path] = "/home/wangshenyu/application/paper_fuzzing/attack_server/modbus_attack_server/ethercat/log_data_communications/logfirst_" + file_name.replace('.txt', '')
    elif os.path.isfile(train_data_path):
        files_dict[train_data_path] = "/home/wangshenyu/application/paper_fuzzing/attack_server/modbus_attack_server/ethercat/log_data_communications/logfirst_" + time.strftime("%H.%M.%S#%Y-%m-%d",
                                                                                 time.localtime())
        print("it's a normal file")
    else:
        print("it's a special file(socket,FIFO,device file)")
        exit()


    start_time = time.strftime("%H.%M.%S#%Y-%m-%d",time.localtime())
    for file_path in files_dict.keys():
        fatal_error_count = 0
        conventional_error_count = 0
        fatal_error_list = []
        conventional_error_list = []
        log_file_name = files_dict[file_path]
        # print("tested file path : " + file_path + "\n" + "log file name : " + log_file_name)

        #
        if not os.path.exists(log_file_name):
            os.mkdir(log_file_name)

        fatal_error_file = log_file_name + "/fatal_error.txt"
        conventional_error_file = log_file_name + "/conventional_error.txt"
        ac_file = log_file_name + "/ac.txt"
        manifest_file = log_file_name + "/manifest.txt"

        f_fatal_log = open(fatal_error_file,
                  "w+")
        f_conventional_log = open(conventional_error_file,
                  "w+")
        f_ac_log = open(ac_file,
                  "w+")
        f_manifest_log = open(manifest_file,
                  "w+")

        try:
            with open(file_path, 'r') as f:
                try:
                    # print("1111111111111111")
                    ECAT_collect2server()
                    
                    content = f.readlines()
                    sum_count = len(content)
                    if seek_number >= sum_count:
                        seek_number = sum_count
                    else:
                        sum_count = seek_number
                    for i in tqdm(range(seek_number)):
                        val = content[i]
                        if val is not None and val != '\n':
                            val = val.strip()
                            string = dataSwitch(val.strip('\n'))    # switch hex to bit for sending it to the simulations
                            try:
                                ethercat_sock.send(string, EthernetSocket.ETH_CAT)
                            except Exception as e:
                                ECAT_collect2server()
                                ethercat_sock.send(string, EthernetSocket.ETH_CAT)
                            f_manifest_log.write(" No." +str(i) + ' TX :' + val)

                            time.sleep(0.1)
                            try:
                                data = ethercat_sock.recv(BUFFER_SIZE, EthernetSocket.ETH_CAT)
                            except IOError as e:
                        

                                fatal_error_count += 1
                                fatal_error_list.append(val)
                                f_manifest_log.write('\n Fatal error! ########################################## \n')
                                f_manifest_log.write('\n Exception: '+str(e) + '\n')
                                f_manifest_log.write('\n')
                                print('No. ' + str(i) + " rejected by server; content: " + val)

                                f_fatal_log.write(val + '\n')
                                
                                ECAT_collect2server()
                                continue
                

                            result = data.hex()
                    
                            if result == '' or result is None:
                                conventional_error_count += 1
                                conventional_error_list.append(val)
                                f_manifest_log.write('\n Conventional error! **********************************************  \n')
                                f_manifest_log.write('\n')
                                print('No. ' + str(i) + " Malformed packet; content: " + val)

                                f_conventional_log.write(val + '\n')
                            else:
                                f_manifest_log.write('\n')
                                f_manifest_log.write("==> No." + str(i) + ' RX :' + result + '\n')
                                f_manifest_log.write('\n')
                                f_ac_log.write(val + '\n')

                except Exception as e:
                    print(e)
                finally:
                    if i >= sum_count-1:
                        log_sum = "Total tested packed number : " + str(seek_number) + "\n"\
                            "Fatal error count : " + str(fatal_error_count) + "\nConventional error count :" + str(conventional_error_count) + "\n" \
                                  "Error rate :" + str(fatal_error_count + conventional_error_count) + "/" + str(sum_count) + "; \n" \
                                 "Acceptance(1 - Fatal/Sum) : {:.3%}  \nBug rate(Malformed packet/Sum) :{:.3%}".format(1 - (fatal_error_count / sum_count),  (conventional_error_count / sum_count))
                        print(log_sum)
                        f_manifest_log.write(log_sum)
        except:
            None
        finally:
            f_manifest_log.close()
            f_ac_log.close()
            f_conventional_log.close()
            f_fatal_log.close()

    end_time = time.strftime("%H.%M.%S#%Y-%m-%d",time.localtime())
    with open("/home/wangshenyu/application/paper_fuzzing/attack_server/modbus_attack_server/ethercat/log_data_communications/time.txt","w+") as time_txt:
        time_txt.write("start_time:"+start_time+"\n")
        time_txt.write("end_time:"+end_time+"\n")

