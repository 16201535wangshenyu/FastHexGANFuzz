import time
import subprocess
import os
import signal
import sys
import re

num_frame_has_fuzzing = 0
# 注册信号 SIGUSR1
clients_py_list = ["FrameSenderReciver_Main_Classification1.py","FrameSenderReciver_Main_Classification2.py","FrameSenderReciver_Main_Classification3.py",
                   "FrameSenderReciver_Main_Classification4.py","FrameSenderReciver_Main_Classification5.py"]
server_port = 5683
server_root_folder = r"/home/wangshenyu/application/paper_fuzzing/tools/libcoap-4.3.0"
server_shell_file_path = server_root_folder + os.path.sep + "install/bin/coap-server"

cov_output_folder = r"/home/wangshenyu/application/paper_fuzzing/attack_server/modbus_attack_server/attack_server_result/coap/coap_tcp/seqgan/cov"
cov_filter_folder = r"'*/tests/*' '*/man/*' '*/usr/include/*'"

server_shell_name = "coap-serv" # according by shell: lsof -i:5683

def check_udp_server_process_alive(port_num):
    # port_num = 5683
    text  =None
    try:
        p = subprocess.Popen('lsof -i:' + str(port_num), shell=True,stdout=subprocess.PIPE,encoding='utf-8')
        text = p.communicate()[0]
        
        p.wait()
        if text is None or text.strip() == "":
            return False
        else:
            return True
    except Exception as e:
        print("port_num:",port_num,"text:",text,file=sys.stderr)

def start_server():
    global server_port
    global server_shell_file_path
    cmd  = "ulimit -c 10240 && "+server_shell_file_path
    if not check_udp_server_process_alive(server_port):
        subprocess.Popen(cmd , shell=True,stdout=subprocess.PIPE)
    # time.sleep(1) # 等待服务器启动起来   
"""
获得服务器进程的pid
"""
def get_server_pid(server_shell_name , PORT):
    pp = subprocess.Popen('lsof -i:'+ str(PORT) , shell=True,stdout=subprocess.PIPE,encoding="utf-8")
    content = pp.communicate()[0]
    if content is not None and content.strip(" ").strip("\n") != "":
        content = content.split("\n")
        for cont in content:
            cont = re.sub(' +', ' ', cont) # 将字符串中多个空格变成一个
            cont = cont.split(" ")
            if server_shell_name in cont[0]:
                content = cont[1]    
                break  
            else:  
                content = None
    else:
        content = None

    
    # if content is not None and len(content) >1:
    #     content = content[1]
    # else:
    #     content = None
    if content is not None and content != "":
        return int(content)
    else:
        return None
    
def exit_server(PORT,is_hard = True):
    # 获取udp server 进程id
    # pp = subprocess.Popen('lsof -i:'+ str(PORT) , shell=True,stdout=subprocess.PIPE,encoding="utf-8")
    # content = pp.communicate()[0].split("\n")[1].split(" ")[1].strip()
    global server_shell_name
    server_pid = get_server_pid(server_shell_name,PORT)
    if server_pid is not None and server_pid != "":
        try:    
            if is_hard:
                os.kill(server_pid, signal.SIGUSR2) 
            else:
                os.kill(server_pid, signal.SIGINT) 
        except Exception as e:
            print("exit_server Fail,server_pid:", server_pid)
    # p.kill()
    # p.terminate()
    # p.wait()
    # # os.killpg(p.pid, signal.SIGINT) 
    # print("UDP 服务器程序已关闭")
    # p= None 
"""
SIGUSR1:信号是用来重新启动服务器的
"""
def SIGUSR1_handler(sig_no,frame):
    global server_port
    exit_server(server_port)
    start_server()


"""
获得客户端进程的pid
"""
def get_clients_pid_list(clients_py_list):
    clients_pid = []
    
    for client in  clients_py_list:

        pid1 = os.popen( 'ps -ef | grep '+client+' | grep -v "grep"').read() #获取整个进程的状态，在第一个grep 后输入的是你python脚本的名称
        pid1 = re.sub(' +', ' ', pid1) # 将字符串中多个空格变成一个
        if pid1 is not None and  pid1 != "":
            pid1 = pid1.split("\n")[-2] # 命令每一行都有一个 \n
            pid1 = pid1.split(" ")

        if pid1 is not None and len(pid1) > 1:
            clients_pid.append(int(pid1[1]))
    
        
    return clients_pid


def close_all_client(clients_pid):
    for pid in clients_pid:
        os.kill(pid,signal.SIGUSR1)

def collect_coverage_info(server_root , output_folder,filter_folder,is_enable_branch_cov = True):
    """
    filter_folder "'*/tests/*' '*/man/*'"
    """
    enable_branch_flags = " --rc lcov_branch_coverage=1 "
    if not is_enable_branch_cov:
        enable_branch_flags=""

    cur_time =time.strftime("%H.%M.%S#%Y-%m-%d",time.localtime())
    output_folder = output_folder + os.path.sep + cur_time
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    
    zero_cmd = "lcov -c -i -d . -o app_base.info"
    first_cmd = "lcov -c -d . "+ enable_branch_flags +"-o app_test.info"
    second_cmd = "lcov -a app_base.info -a app_test.info "+ enable_branch_flags +" -o app_total.info"
    if filter_folder is not None:        
        third_cmd = "lcov "+enable_branch_flags+" --remove app_total.info  "+ filter_folder + " -o app_total.info"
    else:
        third_cmd = "echo 'No filter!'"    
    forth_cmd = "genhtml  -o "+ output_folder +" --legend "+ enable_branch_flags +"--title 'lcov' app_total.info"

    pp = subprocess.Popen(zero_cmd + " && "+first_cmd  + " && "+ second_cmd + " && "+ third_cmd + " && "+ forth_cmd, shell=True,stdout=subprocess.PIPE,cwd=server_root)
    pp.wait()

    

def clear_cvoverage_info(server_root ):
    rm_all_gcda_cmd = r"find . -name '*.gcda' -type f -print -exec rm -rf {} \;"
    rm_all_info_cmd = r"rm -rf *.info"
    pp = subprocess.Popen(rm_all_gcda_cmd + " && "+rm_all_info_cmd , shell=True,stdout=subprocess.PIPE,cwd=server_root)
    pp.wait()

def start_collect_coverage_info(clients_py_list,server_root,server_port,cov_output_folder,cov_filter_folder):
    
    # 1: 向三个fuzz进程发送 sigusr2 信号，让三个进程睡眠6s
    # 2：向服务器进程发送sigint信号，让其停下来，没必要让服务器停下来，只需要向服务器注册一个sigusr1信号，让他将当前的覆盖率信息dump出来即可
    # 3：收集覆盖率信息
    global server_shell_name
    # clients_pid = get_clients_pid_list(clients_py_list)
    # for pid in clients_pid:
    #     os.kill(pid,signal.SIGUSR2)

    server_pid = get_server_pid(server_shell_name,server_port)
    # print("start_collect_coverage_info::::server_pid:",server_pid,file=sys.stderr)
    if server_pid is not None:
        os.kill(server_pid,signal.SIGUSR1) 
        # print("start_collect_coverage_info::::os.kill(pid,signal.SIGUSR1)",file=sys.stderr)
    
    collect_coverage_info(server_root,cov_output_folder,cov_filter_folder)
    # start_server() # 服务器当收到SIGUSR1会退出，因此需要重新启动起来
def exit_fuzz():
    global clients_py_list
    global server_port
    global server_root_folder
    clients_pid = get_clients_pid_list(clients_py_list)
    close_all_client(clients_pid)
    while len(get_clients_pid_list(clients_py_list)) !=0:
        pass
    exit_server(server_port,is_hard=False)    
    # clear_cvoverage_info(server_root_folder)
"""
SIGINT:信号是用来退出 fuzzing 的
"""
def sigint_handler(sig_no,frame):
    exit_fuzz()
    start_collect_coverage_info(clients_py_list,server_root_folder,server_port,cov_output_folder,cov_filter_folder)
    clear_cvoverage_info(server_root_folder)
    exit(0)

count = 0
"""
SIGUSR2:信号是用来获取fuzzing 客户端已经fuzzing了多少frame
"""
def SIGUSR2_handler(sig_no,frame):
    global num_frame_has_fuzzing
    global clients_py_list
    global server_port
    global server_root_folder
    global server_shell_file_path
    global cov_output_folder
    global cov_filter_folder
    global count
    num_frame_has_fuzzing = num_frame_has_fuzzing + 1

    
    if num_frame_has_fuzzing % 30000 == 0:
        count = count + 1
        if count < 10:
            start_collect_coverage_info(clients_py_list,server_root_folder,server_port,cov_output_folder,cov_filter_folder)
        

        cur_time =time.strftime("%H.%M.%S#%Y-%m-%d",time.localtime())
        # print("cur_time:",cur_time,file=sys.stderr)
        # print("The program is running ",count," hours!",file=sys.stderr)
        if count >= 10:
            #向三个fuzz进程发送结束信号。
            exit_fuzz()
            while check_udp_server_process_alive(server_port): # 等待服务器退出
                pass
            start_collect_coverage_info(clients_py_list,server_root_folder,server_port,cov_output_folder,cov_filter_folder)
            clear_cvoverage_info(server_root_folder)
        
if __name__ == "__main__":

    signal.signal(signal.SIGUSR1,SIGUSR1_handler)
    signal.signal(signal.SIGUSR2,SIGUSR2_handler)
    signal.signal(signal.SIGINT,sigint_handler)

    clear_cvoverage_info(server_root_folder) # 预先清理一下服务器中 上次fuzz的覆盖率信息
    # start_server()
    time.sleep(24*60*60*1000)


