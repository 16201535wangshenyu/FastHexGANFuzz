
#################################
"""
clustering via the length of bytes 分类
"""
#################################
len_bytes = {}
origin_file_path = r'G:\研究生\研究生\科研\小论文\FCS-21046_Proof_hi\代码2\WGAN\data\coap\third_dataset\generate_frame\original_frame\generate_coap_raw_frame\2023-05-22-21-31-38_300k.txt'
with open(origin_file_path, 'r') as f:
    try:
        content = f.readlines()
        for i, val in enumerate(content):
            if val is not None and val != '\n':
                val = val.strip()
                if len(val) in len_bytes.keys():
                    len_bytes[len(val)].append(val)
                else:
                    len_bytes[len(val)] = []
                    len_bytes[len(val)].append(val)
    except IOError as e:
        f.close()
        print('can not read the file!')
    finally:
        for key in len_bytes:
            path = 'every_len/cases_c2s_5w_len' + str(key) + '.txt'
            fileObject = open(path, 'w+')
            for item in len_bytes[key]:
                fileObject.write(item)
                fileObject.write('\n')
            fileObject.close()
