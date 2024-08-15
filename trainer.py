import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume

def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    # from datasets.dataset_acdc import Synapse_dataset, RandomGenerator      # ACDC gao
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, edge_dir=args.train_eg_dir, list_dir=args.list_dir, split="train",   #add edge_dir gao
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)  #gao

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    # stablenet 0
    criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    # stablenet 0
    Lam = 0.9
    Alp = 1 - Lam
    par = 0.1


    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            # stablenet 1
            pre_features = torch.zeros(args.batch_size, 512).cuda()  # shape: [bs 512] -> [bs 1024]
            pre_weight1 = torch.ones(args.batch_size, 1).cuda()  # shape: [bs 1]
            # stablenet 1


            image_batch, label_batch, edge_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['edge']   #edge gao
            image_batch, label_batch, edge_batch = image_batch.cuda(), label_batch.cuda(), edge_batch.cuda()

            #outputs = model(image_batch)

            #stablenet 2
            outputs, weight1, pre_features, pre_weight1, o4, o3, o2, o1, edge_out = model(image_batch, args, pre_features, pre_weight1, epoch_num, i_batch)                                      # edge_out.shape: [24 1 224 224]
            # loss_sn = criterion(outputs, label_batch[:].long()).view(1, -1).mm(weight1).view(1)
            # print(weight1)
            loss_sn_pre = nn.Linear(50176, 1, bias=True).cuda()(criterion(outputs, label_batch[:].long()).view(weight1.size()[0], -1)).view(1, -1)
            loss_sn = loss_sn_pre.mm(weight1).view(1)

            # 50176 = 224 * 224; loss_sn_pre.shape: [1 24], weight1.shape: [24 1], loss_sn.shape: [1,]

            #edge_loss
            LogitsBCE = torch.nn.BCEWithLogitsLoss()
            loss_edge = LogitsBCE(edge_out, edge_batch.float())

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            loss_ce4 = ce_loss(o4, label_batch[:].long())       # deep supervision gao
            loss_ce3 = ce_loss(o3, label_batch[:].long())
            loss_ce2 = ce_loss(o2, label_batch[:].long())
            loss_ce1 = ce_loss(o1, label_batch[:].long())
            loss_dice4 = dice_loss(o4, label_batch, softmax=True)
            loss_dice3 = dice_loss(o3, label_batch, softmax=True)
            loss_dice2 = dice_loss(o2, label_batch, softmax=True)
            loss_dice1 = dice_loss(o1, label_batch, softmax=True)
            loss_o4 = Alp * loss_ce4 + Lam * loss_dice4
            loss_o3 = Alp * loss_ce3 + Lam * loss_dice3
            loss_o2 = Alp * loss_ce2 + Lam * loss_dice2
            loss_o1 = Alp * loss_ce1 + Lam * loss_dice1
            loss_ex = loss_o1 + loss_o2 + loss_o3 + loss_o4

            loss = Alp * loss_ce + Lam * loss_dice + par * loss_ex + par * loss_sn + par * loss_edge

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"