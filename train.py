import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config

# python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml
# --max_epochs 150 --base_lr 0.05 --batch_size 24
Dataset_name = 'FLARE22'
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    # default='../data/'+Dataset_name, help='root dir for data')
                    default='F:/'+Dataset_name, help='root dir for data')
parser.add_argument('--train_eg_dir',  type=str,
                    # default='../data/'+Dataset_name+'/train_edge')  #edge_path
                    default='F:/'+Dataset_name+'/train_edge', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default=Dataset_name, help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_'+Dataset_name, help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=14, help='output channel of network')
parser.add_argument('--output_dir', type=str,
                    default='./output/FLARE22',                                       # path
                    help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')  # 150
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')  # 24
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,                         # 0.05
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

### stablenet ###
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument ('--lrbl', type = float, default = 1.0, help = 'learning rate of balance')

parser.add_argument('--cos', '--cosine_lr', default=1, type=int,
                    metavar='COS', help='lr decay by decay', dest='cos')
parser.add_argument ('--epochb', type = int, default = 150, help = 'number of epochs to balance')    # 20 should be changed to 150
parser.add_argument('--epochs', default=150, type=int, metavar='N',      # 30 should be changed to 150
                    help='number of total epochs to run')
parser.add_argument ('--epochs_decay', type=list, default=[24, 30], help = 'weight lambda for second order moment loss')

parser.add_argument ('--num_f', type=int, default=6, help = 'number of fourier spaces')    # num_f 5 is best in paper, org is 1
parser.add_argument('--sum', type=bool, default=True, help='sum or concat')

parser.add_argument ('--decay_pow', type=float, default=2, help = 'value of pow for weight decay')
# for expectation
parser.add_argument ('--lambda_decay_rate', type=float, default=1, help = 'ratio of epoch for lambda to decay')
parser.add_argument ('--lambda_decay_epoch', type=int, default=5, help = 'number of epoch for lambda to decay')
parser.add_argument ('--min_lambda_times', type=float, default=0.01, help = 'number of global table levels')

parser.add_argument ('--lambdap', type = float, default = 70.0, help = 'weight decay for weight1 ')

parser.add_argument ('--first_step_cons', type=float, default=1, help = 'constrain the weight at the first step')

parser.add_argument('--presave_ratio', type=float, default=0.9, help='the ratio for presaving features')
### stablenet ###

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
if args.dataset == "ACDC":
    args.root_path = os.path.join(args.root_path, "train")
config = get_config(args)
if args.dataset == "AMOS":
    args.root_path = os.path.join(args.root_path, "train_npz_new")
if args.dataset == "FLARE22":
    args.root_path = os.path.join(args.root_path, "train")

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'train_eg_dir': args.train_eg_dir,   #edge gao
        },
        'ACDC': {
            'root_path': args.root_path,  # ../data/Synapse/train_npz
            'list_dir': './lists/lists_ACDC',  # ./lists/lists_Synapse
            'num_classes': 4,
            'train_eg_dir': args.train_eg_dir,
        },
        'AMOS': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_AMOS',
            'num_classes': 16,
            'train_eg_dir': args.train_eg_dir,  # edge gao
        },
        'FLARE22': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_FLARE22',
            'num_classes': 14,
            'train_eg_dir': args.train_eg_dir,  # edge gao
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.train_eg_dir = dataset_config[dataset_name]['train_eg_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_from(config)

    trainer = {Dataset_name: trainer_synapse,}
    trainer[dataset_name](args, net, args.output_dir)