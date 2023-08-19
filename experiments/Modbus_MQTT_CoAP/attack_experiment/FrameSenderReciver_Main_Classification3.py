import socket
import struct
import time
import os
from tqdm import tqdm
import subprocess
import signal
import re
import random
""" 
#####   执行目录为 ..\SAGAN_MQTT; 
报文的功能全部为发布消息的报文
"""
connect_message_data = []
cur_connect_message_data_index = 0
connect_message_data_file_list = []
connect_message_data_file_list_index = 0

s = None
TCP_IP  = None
TCP_PORT = None
close_flag = None

"""
得到监控进程的PID
"""
def get_monitor_process_pid():
    monitor_py = r"monitor_process.py"
    pid1 = os.popen( 'ps -ef | grep '+monitor_py+' | grep -v "grep"').read() #获取整个进程的状态，在第一个grep 后输入的是你python脚本的名称
    pid1 = re.sub(' +', ' ', pid1) # 将字符串中多个空格变成一个
    if pid1 is not None and  pid1 != "":
        pid1 = pid1.split("\n")[-2] # 命令每一行都有一个 \n
        pid1 = pid1.split(" ")
    if pid1 is not None and len(pid1) > 1:
        return int(pid1[1])
    else:
        return None
def start_coap_tcp_server():
    global close_flag
    monitor_process_pid = get_monitor_process_pid()
    if monitor_process_pid is None:
        close_flag = True
    else:
        os.kill(monitor_process_pid,signal.SIGUSR1)
        time.sleep(1) #等待服务器启动
def tell_monitor_process_fuzzing_one_frame():
    global close_flag
    monitor_process_pid = get_monitor_process_pid()
    if monitor_process_pid is None:
        close_flag = True
    else:
        os.kill(monitor_process_pid,signal.SIGUSR2)
        
def check_tcp_server_process_alive():
    global TCP_PORT
    p = subprocess.Popen('lsof -i:' + str(TCP_PORT), shell=True,stdout=subprocess.PIPE,encoding='utf-8')
    text = p.communicate()[0]
    p.wait()
    if text is None or text.strip() == "":
        return False
    else:
        return True
    
def refresh_connect_message_data():
    global connect_message_data_file_list
    global connect_message_data_file_list_index
    global connect_message_data
    global cur_connect_message_data_index
    if connect_message_data_file_list_index >=len(connect_message_data_file_list):
        connect_message_data.clear()
        cur_connect_message_data_index = 0
        return
    


    connect_message_data.clear()
    with open(connect_message_data_file_list[connect_message_data_file_list_index],"r") as file:
        file_content = file.readlines()
        connect_message_data.extend(file_content)

    cur_connect_message_data_index = 0
    connect_message_data_file_list_index = connect_message_data_file_list_index + 1

def dataSwitch(data):
    str1 = ''
    str2 = b''#############################################################################################################
    while data:
        str1 = data[0:2]
        s = int(str1,16)
        str2 += struct.pack('B',s)
        data = data[2:]
    return str2

"""max_index:最大递归次数"""
max_dfs_num = 6
def TCP_collect2server(max_index,f_manifest_log):
    global max_dfs_num 
    global TCP_IP
    global TCP_PORT

    global s
    is_hang_out = False
    if not check_tcp_server_process_alive() or max_index == max_dfs_num:
        
        if max_index == max_dfs_num:
            print("hang out!")
            f_manifest_log.write("HANG_OUT!\n")
            is_hang_out = True
        if not start_coap_tcp_server():
            print("coap-server exit() and cannot start by monitor.py")
            return {
                "result":False,
                "is_hang_out":is_hang_out
            }
    
    try:     
        # time.sleep(1)
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.setblocking(0)
        s.settimeout(0.5) # 设置接收函数 多少秒接收不到会断开连接
        s.connect((TCP_IP, TCP_PORT))
        return {
            "result": True,
            "is_hang_out":is_hang_out
        }
    except Exception as e:
        s.close()
        TCP_collect2server(max_index+1,f_manifest_log)


# def COAP_collect2server():
    
#     global s
#     global cur_connect_message_data_index
#     global connect_message_data
#     req_val = ""
#     try:
#         if s is not None:
#             s.close()
#             s = None
#         if not TCP_collect2server(0): # 如果服务器已经停止，而且监控程序也下线了
#             return 
#         # time.sleep(0.1)
        
#         # if connect_message_data and connect_message_data[cur_connect_message_data_index] is not None and connect_message_data[cur_connect_message_data_index] != "\n":
#         #     req_val = connect_message_data[cur_connect_message_data_index]
#         # else:
#         #     req_val = "50e12380010020"

#         # if max_index >=max_dfs_num:
#         #     req_val = "50e12380010020"
#         # else:
#         #     cur_connect_message_data_index  = cur_connect_message_data_index + 1

#         # if cur_connect_message_data_index >= len(connect_message_data):
#         #     # connect_message_data.clear()
#         #     refresh_connect_message_data()

#         # req_string = dataSwitch(req_val.strip('\n'))  # switch hex to bit for sending it to the simulations
#         # s.send(req_string)
#         # time.sleep(0.1)
#         # req_ack = s.recv(BUFFER_SIZE).hex()  # 获取连接请求回应数据包
        
#     except Exception as e:
#         # print(" rejected by server; content: " + req_val,"Excetion:",e)
#         # if max_index >=max_dfs_num:
#         print(e)
#         # COAP_collect2server(max_index+1)

'''
    针对连接重置异常，导致数据包没有发送出去，则进行重发
'''
def resend_message(massage,f_manifest_log):
    global s
    try:
        result = TCP_collect2server(0,f_manifest_log)
        if result["result"]:
            s.send(massage)
    except Exception as e:
        resend_message(massage,f_manifest_log)
"""
SIGUSR1:信号是用来接收 监控进程 发送的SIGUSR1,以结束fuzz进程
"""
def SIGUSR1_handler(sig_num,frame):
    global close_flag
    close_flag = True

"""
SIGUSR2:信号是用来接收 监控进程 发送的SIGUSR2,以将fuzz进程睡眠,让监控进程进行覆盖率计算
"""
def SIGUSR2_handler(sig_num,frame):
    time.sleep(0.1)

if __name__ == '__main__':
    signal.signal(signal.SIGUSR1,SIGUSR1_handler)
    signal.signal(signal.SIGUSR2,SIGUSR2_handler)
    TCP_IP = '::1'
    TCP_PORT = 5683
    # time.sleep(3) #等待监控进程启动起来
    close_flag = False
    if not check_tcp_server_process_alive():
        start_coap_tcp_server()
    
    train_data_path = r'/home/wangshenyu/application/paper_fuzzing/attack_server/modbus_attack_server/attack_server_result/coap/coap_tcp/seqgan/in/frame_fuzzing3'
    # train_data_path_connect = r"/home/wangshenyu/application/paper_fuzzing/attack_server/modbus_attack_server/attack_server_result/coap/test/test2/connect"
    output_data_path = r"/home/wangshenyu/application/paper_fuzzing/attack_server/modbus_attack_server/attack_server_result/coap/coap_tcp/seqgan/log_data_communications"

    # connect_message_data_file_list = os.listdir(train_data_path_connect)
    # for i in range(len(connect_message_data_file_list)):
    #     connect_message_data_file_list[i] = train_data_path_connect + os.path.sep + connect_message_data_file_list[i]
    # random.shuffle(connect_message_data_file_list)
    # refresh_connect_message_data()
    
    
    seek_number = 3 * 10000

    global sum_count
    global BUFFER_SIZE



    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # global s
    BUFFER_SIZE = 100000

    files_dict = {}
    if os.path.isdir(train_data_path):
        print("it's a directory")
        for file_name in os.listdir(train_data_path):
            file_path = train_data_path + "/" + file_name
            files_dict[file_path] = output_data_path+ "/logfirst_" + file_name.replace('.txt', '')
    elif os.path.isfile(train_data_path):
        files_dict[train_data_path] = output_data_path+ "/logfirst_" + time.strftime("%H.%M.%S#%Y-%m-%d",
                                                                                 time.localtime())
        print("it's a normal file")
    else:
        print("it's a special file(socket,FIFO,device file)")
        exit()

    start_time =time.strftime("%H.%M.%S#%Y-%m-%d",time.localtime())
    for file_path in files_dict.keys():
        if close_flag:
            break

        fatal_error_count = 0
        conventional_error_count = 0
        fatal_error_list = []
        conventional_error_list = []
        log_file_name = files_dict[file_path]
        print("tested file path : " + file_path + "\n" + "log file name : " + log_file_name)

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
                    TCP_collect2server(0,f_manifest_log)
                    content = f.readlines()
                    random.shuffle(content)
                    sum_count = len(content)
                    if seek_number >= sum_count:
                        seek_number = sum_count
                    else:
                        sum_count = seek_number
                    for i in tqdm(range(seek_number)):
                        
                        if close_flag:
                            break    

                        val = content[i]
                        if val is not None and val != '\n':
                            val = val.strip()
                            string = dataSwitch(val.strip('\n'))    # switch hex to bit for sending it to the simulations
                            try:
                                s.send(string)
                            except Exception as e:
                                if not check_tcp_server_process_alive():
                                    start_coap_tcp_server()
                                    f_manifest_log.write(" No." +str(i) + ' : CORE DUMP\n')
                                resend_message(string,f_manifest_log)
                            
                            tell_monitor_process_fuzzing_one_frame()

                            f_manifest_log.write(" No." +str(i) + ' TX :' + val)
                            time.sleep(0.1)
                            try:
                                data = s.recv(BUFFER_SIZE)
                            except IOError as e:
                        

                                fatal_error_count += 1
                                fatal_error_list.append(val)
                                f_manifest_log.write('\n Fatal error! ########################################## \n')
                                f_manifest_log.write(' Exception: '+str(e) + '\n')
                                
                                print('No. ' + str(i) + " rejected by server; content: " + val)    
                                f_fatal_log.write(val + '\n')
                                # 检查UDP服务器进程是否已经消失
                                if not check_tcp_server_process_alive():
                                    cur_time =time.strftime("%H.%M.%S#%Y-%m-%d",time.localtime())
                                    f_manifest_log.write(' CORE DUMP: '+cur_time + '\n')
                                    # break
                                    start_coap_tcp_server()
                
                                    
                                f_manifest_log.write('\n')
                                TCP_collect2server(0,f_manifest_log) # 重新建立连接
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
            # exit_coap_udp_server()
    end_time =time.strftime("%H.%M.%S#%Y-%m-%d",time.localtime())

    with open(output_data_path+os.path.sep +"time.txt","w+") as time_f:
        time_f.write("start_time:"+start_time  +"\n" )
        time_f.write("end_time:"+end_time )