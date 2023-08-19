

def cal_SGTC(model_time):
    model_SGTC = {}
    for tool, time_info in model_time.items():
        minutes = (5 * 60 + time_info["exceed_seconds"])/60
        SGTC = time_info["total_frame_num"]/minutes
        model_SGTC[tool] =round(SGTC)
    return model_SGTC

if __name__ == "__main__":
    ### 下面是5分钟生成数据帧的结果
    model_time = {
        "Fast_RoPEGAN_model": {
            "total_frame_num": 2137408,
            "exceed_seconds": 0.005452394485473633,

        },
        "Fast_RoPEGAN_model_without_FastRoPEAttention": {
            "total_frame_num": 1896384,
            "exceed_seconds": 0.006765127182006836,

        },
        "blstm_model": {
            "total_frame_num": 190080,
            "exceed_seconds": 0.006131649017333984,
        },
        "wgan_model": {
            "total_frame_num": 9630464,
            "exceed_seconds": 0.0011548995971679688,
        },
        "lstm_model": {
            "total_frame_num": 71680,
            "exceed_seconds": 0.2472209930419922,
        },
        "seq_gan_model": {
            "total_frame_num": 124736,
            "exceed_seconds": 0.01498723030090332,
        }

    }

    print(cal_SGTC(model_time))
