#MultiTrans-Concat
ver = 1
sver =1
n_class = 20

#fusionType = "condattcat_gpt2_norelu"
use_cam = [0,1,2,3,4,5]
use_mic = [0,1,2,3,4,5,6,7,8,9,10,11]

sp_param = {"sp_cfg": { "fs": 16000.0,
                        "fft_len": 400,
                        "win_len": 400,
                        "shift_len": 160,
                        "input_len": 25.6,
                        "n_mel": 64,
                        "f_min": 0.0,
                        "t_reso": 32}}

dnn_param = {"dnn_cfg": {"n_class":n_class,
                        "use_mic": use_mic,
                        "use_cam": use_cam}}

train_cfg = { "batch_num": 4,         
             "MAX_EPOCH": 200,  
            "opt_level": "O0",#FP32
            "dataType": "multi",
            "vtraindir": "MM_Office_Dataset/video/train",
            "atraindir": "MM_Office_Dataset/audio/train",
            "vtestdir": "MM_Office_Dataset/video/test",
            "atestdir": "MM_Office_Dataset/audio/test",
            "rec_info" : "MM_Office_Dataset/label/trainlabel/recinfo.csv",
            "event_info" : "MM_Office_Dataset/label/trainlabel/eventinfo.csv",
            "testlist_path": "testlist.csv",
            "testlabeldir" : "MM_Office_Dataset/label/testlabel/"}

test_cfg = { "batch_num": 8,       
             "MAX_EPOCH": 200,  
            "opt_level": "O0",#FP32
            "dataType": "multi",
            "divdata": True,
            "testlist_path": "testlist.csv",
            "vvaliddir": "MM_Office_Dataset/video/test",
            "avaliddir": "MM_Office_Dataset/audio/test",
            "testlabeldir" : "MM_Office_Dataset/label/testlabel/"}

train_cfg.update(sp_param)
train_cfg.update(dnn_param)
test_cfg.update(sp_param)
test_cfg.update(dnn_param)


