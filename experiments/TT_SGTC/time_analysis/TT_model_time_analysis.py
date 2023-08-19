
def parse_time(time_str):
    """
    parse time to seconds
    """
    if time_str is None:
        return 0

    # time_str = ""
    time_arr = time_str.split(":")
    hours = int(time_arr[0])
    minutes = int(time_arr[1])
    seconds = int(time_arr[2].split(".")[0])
    return hours * 60 *60 + minutes * 60 + seconds


def TT_parse_model_time(model_time):
    TT_Fast_RoPEGAN_model = {}
    for model_str ,time_dict in model_time.items():

        pre_time_sec = parse_time(time_dict["pre_time"])
        train_time_sec = parse_time(time_dict["train_time"])


        TT_Fast_RoPEGAN_model[model_str] = (pre_time_sec + train_time_sec) / 60 /60

    return TT_Fast_RoPEGAN_model

def TT_L_parse_model_time(mode_time):
    TT_L_model_time = {}
    for model_str, time_dict in model_time.items():
        time_list = []
        for time in time_dict:
            time_list.append(parse_time(time)/60/60)
        TT_L_model_time[model_str] = time_list
    return TT_L_model_time

if __name__ == "__main__":
    # model_time = {
    #     "Fast_RoPEGAN_model": {
    #         "pre_time": None,
    #         "train_time": "2:13:43.967686",
    #
    #     },
    #     "Fast_RoPEGAN_model_without_FastRoPEAttention": {
    #         "pre_time": None,
    #         "train_time": "2:25:52.301827",
    #
    #     },
    #     "blstm_model": {
    #         "pre_time": "1:55:28.051763",
    #         "train_time": "22:15:56.021186",
    #     },
    #     "wgan_model": {
    #         "pre_time": None,
    #         "train_time": "0:12:47.889376",
    #     },
    #     "lstm_model": {
    #         "pre_time": None,
    #         "train_time": "6:59:44.321855",
    #     },
    #     "seq_gan_model": {
    #         "pre_time": None,
    #         "train_time": "27:44:58.672609",
    #     }
    #
    # }
    #
    #
    # print(TT_parse_model_time(model_time))
############################################################################################
    model_time = {
        "Fast_RoPEGAN_model": [
            "2:13:43.967686",
            "2:18:32.983268",
            "2:23:47.311337",
            "2:29:37.860628",
            "2:35:05.973285",
            "2:39:58.537831",
            "2:46:41.004606",
            "2:53:19.721843",
            "2:58:46.123106",
            "3:08:25.387948",
        ],
        "Fast_RoPEGAN_model_without_FastRoPEAttention": [
            "2:25:52.301827",
            "2:35:33.084097",
            "2:45:54.701644",
            "2:56:15.823319",
            "3:07:15.911812",
            "3:18:49.376044",
            "3:31:36.333705",
            "3:50:46.634255",
            "4:04:07.263062",
            "4:25:17.265076",

        ]
    }
    print(TT_L_parse_model_time(model_time))
