class Config_UCLM:
    # This dataset is for breast cancer segmentation
    data_path = "/mnt/HDD1/tuong/TRUST/dataset"
    train_path = "/mnt/HDD1/tuong/TRUST/dataset/Breast-UCLM/train.txt"
    val_path = "/mnt/HDD1/tuong/TRUST/dataset/Breast-UCLM/valid.txt"
    test_path = "/mnt/HDD1/tuong/TRUST/dataset/Breast-UDIAT/valid.txt"
    save_path = "./checkpoints2/Breast-UCLM/"
    result_path = "./result/Breast-UCLM/"
    tensorboard_path = "./tensorboard/Breast-UCLM/"
    load_path = save_path + "/SAMUS_best.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 0.001                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_BUSI:
    # This dataset is for breast cancer segmentation
    data_path = "/mnt/HDD1/tuong/TRUST/dataset"
    train_path = "/mnt/HDD1/tuong/TRUST/dataset/Breast-BUSI/train.txt"
    val_path = "/mnt/HDD1/tuong/TRUST/dataset/Breast-BUSI/valid.txt"
    test_path = "/mnt/HDD1/tuong/TRUST/dataset/Breast-UDIAT/valid.txt"
    save_path = "./checkpoints2/Breast-BUSI/"
    result_path = "./result/Breast-BUSI/"
    tensorboard_path = "./tensorboard/Breast-BUSI/"
    load_path = save_path + "/SAMUS_best.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 0.001                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_UDIAT:
    # This dataset is for breast cancer segmentation
    data_path = "/mnt/HDD1/tuong/TRUST/dataset"
    train_path = "/mnt/HDD1/tuong/TRUST/dataset/Breast-UDIAT/train.txt"
    val_path = "/mnt/HDD1/tuong/TRUST/dataset/Breast-UDIAT/valid.txt"
    test_path = "/mnt/HDD1/tuong/TRUST/dataset/Breast-BUSI/valid.txt"
    save_path = "./checkpoints2/Breast-UDIAT/"
    result_path = "./result/Breast-UDIAT/"
    tensorboard_path = "./tensorboard/Breast-UDIAT/"
    load_path = save_path + "/SAMUS_best.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 100                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 0.001                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

# ==================================================================================================
def get_config(task="US30K"):
    if task == "BUSI":
        return Config_BUSI()
    elif task == "UCLM":
        return Config_UCLM()
    elif task == "UDIAT":
        return Config_UDIAT()
    else:
        assert("We do not have the related dataset, please choose another task.")