import math
import argparse
import torch
from torch.nn import DataParallel
from prediction.utils_for_CLSTM_prediction.metrics import *
from prediction.utils_for_CLSTM_prediction.loaddata import getTestingData
from models_CLTSM import net, modules
from models_CLTSM.backbone_dict import backbone_dict
from prediction.utils_for_CLSTM_prediction.functions_for_prediction import *


# ***********************************************************************************Training settings
parser = argparse.ArgumentParser(description='models on depth data')
parser.add_argument('--data_root_dir', type=str, default='/home/hkzhang/phoenix_fast/Depth_estimation/data/nyu_v2_r')
parser.add_argument('--data_list_root', type=str, default='../data/data_list/')
parser.add_argument('--dataset', type=str, default='raw_nyu_v2_250k')
parser.add_argument('--backbone', type=str, default='resnet18')
parser.add_argument('--refinenet', type=str, default='R_CLSTM_5')
parser.add_argument('--use_gan', type=bool, default=True)
parser.add_argument('--test_loc', type=str, default='end')
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--fl', type=int, default=5)
parser.add_argument('--overlap', type=int, default=0)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--devices', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--trained_models_dir', type=str, default='./trained_models/')
parser.add_argument('--results_save_dir', type=str, default='./predicton_results/')
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
backbone = backbone_dict[args.backbone]()
Encoder = modules.E_resnet(backbone)
if args.backbone in ['resnet50']:
    model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048], refinenet=args.refinenet)
elif args.backbone in ['resnet18', 'resnet34']:
    model = net.model(Encoder, num_features=512, block_channel=[64, 128, 256, 512], refinenet=args.refinenet)
model = DataParallel(model).cuda()

if not args.use_gan:
    trained_model_dir = args.trained_models_dir + '{}_{}_fl{}.pkl'.format(args.backbone, args.refinenet, args.fl)
else:
    trained_model_dir = args.trained_models_dir + '{}_{}_fl{}_gan.pkl'.format(args.backbone, args.refinenet, args.fl)


model.load_state_dict(torch.load(trained_model_dir))

test_data_dict_dir = args.data_list_root + '{}/{}_fps{}_fl{}_op{}_{}_test.json'.format(args.dataset, args.dataset, args.fps, args.fl, args.overlap, args.test_loc)
test_loader = getTestingData(1, test_data_dict_dir, args.data_root_dir, args.num_workers)

metrics = metric_list([REL(),
                       RMS(),
                       log10(),
                       deta(metric_name='deta1', threshold=1.25),
                       deta(metric_name='deta2', threshold=math.pow(1.25, 2)),
                       deta(metric_name='deta3', threshold=math.pow(1.25, 3))
                       ])

inference(model, test_loader, device, metrics)

