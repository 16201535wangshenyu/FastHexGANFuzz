import os
import operator
import random
'''

每一个文件都需要写入totoal_num/file_num 条frame，如果文件中的记录不够这么多，那么反复写直到够为止
'''

def cluster_write_to_file(output_file,num,file_names):

    with open(output_file, 'w+') as f_w:
        for file in file_names:
            cnt = 0
            file_content = []
            with open(file, 'r') as r_f:
                r_file_content = r_f.readlines()
                for line in reversed(r_file_content):
                    f_w.write(line)
                    file_content.append(line)
                    cnt += 1
                    if cnt == num:
                        break
                # random.shuffle(file_content)
                while cnt != num:
                    for line_ in file_content:
                        cnt += 1
                        f_w.write(line_)
                        if cnt == num:
                            break

def cluster(per_file_frame_num,file_info,allow_max_span):
    start_len  = file_info[0][1]
    file_names = []
    file_lens = []
    count = 0
    for file in file_info:
        count += 1
        value = file[1]
        if value - start_len > allow_max_span or count == len(file_info):
            if count == len(file_info):
                file_names.append(file[0])
                file_lens.append(value)
            start_len = value
            output_file = "training_set/coap_"
            for v in file_lens:
                output_file += str(v) + "_"
            output_file = output_file +str((per_file_frame_num/10000)) +"w.txt"

            cluster_write_to_file(output_file,per_file_frame_num//len(file_names),file_names)
            file_names.clear()
            file_lens.clear()

        file_names.append(file[0])
        file_lens.append(value)

def get_len_by_filename(filename):
    # filename = "cases_c2s_5w_len22.txt"
    prefix = "cases_c2s_5w_len"
    suffix = ".txt"
    filename = filename [len(prefix): (-1*len(suffix))]
    # filename = ""

    num = int(filename)
    return num

def get_all_file_and_len(folder):
    files_list = os.listdir(folder)
    file_dict = {}
    for f in files_list:
        f_len = get_len_by_filename(f)
        file_dict[folder+os.path.sep+ f] = f_len
    file_dict_items = sorted(file_dict.items(),key=operator.itemgetter(1))

    return file_dict_items
if __name__ == "__main__":


    # src_folder = r"every_len"
    # file_names = get_all_file_and_len(src_folder)
    file_names = {
        r'every_len/cases_c2s_5w_len36.txt': 36,
        r'every_len/cases_c2s_5w_len38.txt': 38,
        r'every_len/cases_c2s_5w_len40.txt': 40,
        r'every_len/cases_c2s_5w_len42.txt': 42,
        r'every_len/cases_c2s_5w_len68.txt': 68,
        r'every_len/cases_c2s_5w_len70.txt': 70,
        r'every_len/cases_c2s_5w_len72.txt': 72,
        r'every_len/cases_c2s_5w_len74.txt': 74,
        r'every_len/cases_c2s_5w_len100.txt': 100,
        r'every_len/cases_c2s_5w_len102.txt': 102,
        r'every_len/cases_c2s_5w_len104.txt': 104,
        r'every_len/cases_c2s_5w_len106.txt': 106,
        r'every_len/cases_c2s_5w_len116.txt': 116,
        r'every_len/cases_c2s_5w_len118.txt': 118,
        r'every_len/cases_c2s_5w_len120.txt': 120,
        r'every_len/cases_c2s_5w_len122.txt': 122,

        r'every_len/cases_c2s_5w_len132.txt': 132,
        r'every_len/cases_c2s_5w_len134.txt': 134,
        r'every_len/cases_c2s_5w_len136.txt': 136,
        r'every_len/cases_c2s_5w_len138.txt': 138,
    }

    file_names = sorted(file_names.items(), key=operator.itemgetter(1))

    allow_max_span = 6
    totoal_num = 6 * 10000

    cluster(totoal_num,file_names,allow_max_span)
    # total_len_files_num = len(file_names)
    # num = totoal_num // total_len_files_num
    # output_file = 'mqtt/mqtt_all_lens_13w.txt'