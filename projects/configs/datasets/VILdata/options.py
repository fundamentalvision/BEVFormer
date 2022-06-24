from easydict import EasyDict

OPTION = EasyDict()

# ------------------------------------------ data configuration ---------------------------------------------
OPTION.trainset = ['VIL100']
OPTION.valset = 'VIL100'
OPTION.setting = '60_lr0.001deay1e-6_sgd'
OPTION.root = './dataset'  # dataset root path
OPTION.datafreq = [3] #
OPTION.max_object = 8 # max number of instances
OPTION.input_size = (240, 427)   # input image sizee
OPTION.sampled_frames = 9       # min sampled time length while trianing
OPTION.max_skip = [5]         # max skip time length while trianing
OPTION.samples_per_video = 2  # sample numbers per video

# ----------------------------------------- model configuration ---------------------------------------------
OPTION.keydim = 128
OPTION.valdim = 512
OPTION.save_freq = 5
OPTION.save_freq_max = 100
OPTION.epochs_per_increment = 2

# ---------------------------------------- training configuration -------------------------------------------
OPTION.epochs = 60
OPTION.train_batch = 1
OPTION.learning_rate = 0.001
OPTION.gamma = 0.1
OPTION.momentum = (0.9, 0.999)
OPTION.solver = 'sgd'             # 'sgd' or 'adam'
OPTION.weight_decay = 1e-6  #5e-4
OPTION.iter_size = 1
OPTION.milestone = []              # epochs to degrades the learning rate
OPTION.loss = 'both'               # 'ce' or 'iou' or 'both'
OPTION.mode = 'recurrent'          # 'mask' or 'recurrent' or 'threshold'
OPTION.iou_threshold = 0.65        # used only for 'threshold' training

# ---------------------------------------- testing configuration --------------------------------------------
OPTION.epoch_per_test = 5

# ------------------------------------------- other configuration -------------------------------------------
OPTION.checkpoint = 'models'
OPTION.initial = ''      # path to initialize the backbone
OPTION.initial_STM = ''      # path to initialize the backbone ./models/initial_STM.pth.tar
OPTION.resume_ATT = './models/VIL100/60_lr0.001deay1e-6_sgd/ATT/recurrent35.pth.tar' # path to restart from the checkpoint
OPTION.resume_STM = './models/VIL100/60_lr0.001deay1e-6_sgd/STM/recurrent35.pth.tar' # path to restart from the checkpoint
OPTION.gpu_id = '1'      # defualt gpu-id (if not specified in cmd)
OPTION.workers = 1
OPTION.save_indexed_format = True # set True to save indexed format png file, otherwise segmentation with original image
OPTION.output_dir = 'output'
