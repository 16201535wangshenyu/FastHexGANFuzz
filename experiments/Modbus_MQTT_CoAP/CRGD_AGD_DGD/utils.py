import os.path


def convert_bin_files_to_str_list(bin_folder):
    bin_files = os.listdir(bin_folder)
    str_list = []
    for bin_file in bin_files:
        bin_file_path = bin_folder + os.path.sep + bin_file
        with open(bin_file_path, "rb") as b_f:
            contents = b_f.read()
            contents = contents.hex()
            if len(contents) % 2 != 0:
                print("bad things!")
            str_list.append(contents + "\n")
            # print()
    return str_list

def convert_all_protocol_seeds_to_txt(protocols_seeds_folder, output_folder):
    """
    protocols_seeds_folder = {
    "mqtt" :"bin_seeds_folder_path",
    "coap" :"bin_seeds_folder_path",
    "modbus" :"bin_seeds_folder_path"
}
    """
    for key, value in protocols_seeds_folder.items():
        with open(output_folder + os.path.sep + key + ".txt", "w") as f:
            f.writelines(convert_bin_files_to_str_list(value))


if __name__ == "__main__":
    pass
    # protocols_seeds_folder = {
    #     "mqtt": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\mqtt\afl++\tools_seed",
    #     "coap": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\coap\afl++\tools_seed",
    #     "modbus": r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\modbus\afl++\tools_seed"
    # }
    # output_folder = r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\experiment_analysis\frame_analysis\tools_seed"
    # convert_all_protocol_seeds_to_txt(protocols_seeds_folder,output_folder)


    # a= ""
    # a.startswith()

    # convert_bin_files_to_str_list(
    #     r"D:\code_projects\pycharm\paper_fuzzing_code\attack_experiment\modbus\afl++\tools_seed")
