import datetime
import os
import sys

import torch
from torch import nn
from args import args
from data.Data import CIFAR10, CIFAR100
from resnet_kd import resnet20
from trainer.trainer import validate, train
from utils.utils import set_gpu, get_logger, Logger, set_random_seed
from vgg_kd import cvgg11_bn, cvgg11_bn_small

def main():
    print(args)
    sys.stdout = Logger('print process.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)


def main_worker(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('pretrained_model/' + args.arch_s + '/' + args.set):
        os.makedirs('pretrained_model/' + args.arch_s + '/' + args.set, exist_ok=True)
    logger = get_logger('pretrained_model/' + args.arch_s + '/' + args.set + '/logger' + now + '.log')
    logger.info(args.arch)
    logger.info(args.arch_s)
    logger.info(args.set)
    logger.info(args.batch_size)
    logger.info(args.weight_decay)
    logger.info(args.lr)
    logger.info(args.epochs)
    logger.info(args.lr_decay_step)
    logger.info(args.num_classes)

    if args.arch_s == 'cvgg11_bn_small':
        mask = torch.load('/public/ly/xianyu/pretrained_model/resnet56/cifar100/cifar100_T_resnet56_S_cvgg11_bn_mask.pt')  # 要手动调整
        print(mask['layer_num'])

        model = cvgg11_bn(finding_masks=True, num_classes=args.num_classes, batch_norm=True)
        model_s = cvgg11_bn_small(finding_masks=False, num_classes=args.num_classes, batch_norm=True)
        ckpt = torch.load('/public/ly/xianyu/pretrained_model/resnet56/cifar100/cifar100_cvgg11_bn.pt')  # 要手动调整
        model.load_state_dict(ckpt)
        model_s.classifier[1] = nn.Linear(mask['layer_num'][-1], 512)

    if args.arch_s == 'resnet20_small':
        in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]  # 第一层不减
        out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
        in_cfg_s = [3, 16, 11, 14, 24, 25, 30, 61, 64, 53]  # 要手动调整
        out_cfg_s = [16, 11, 14, 24, 25, 30, 61, 64, 53, 44]

        model = resnet20(finding_masks=True, in_cfg=in_cfg, out_cfg=out_cfg, num_classes=args.num_classes)
        ckpt = torch.load('/public/ly/xianyu/pretrained_model/cvgg16_bn/cifar100/cifar100_resnet20.pt', map_location='cuda:%d' % args.gpu)  # 要手动调整
        model.load_state_dict(ckpt)
        model_s = resnet20(finding_masks=False, in_cfg=in_cfg_s, out_cfg=out_cfg_s, num_classes=args.num_classes)

    model_s = set_gpu(args, model_s)
    model = set_gpu(args, model)
    print(model_s)

    criterion = nn.CrossEntropyLoss().cuda()
    data = CIFAR100()
    acc1, acc5 = validate(data.val_loader, model, criterion, args)
    print(acc1)

    if args.arch_s == 'cvgg11_bn_small':
        load_vgg_model(model_s, model.state_dict())

    if args.arch_s == 'resnet20_small':
        load_resnet_model(model_s, model.state_dict(), layer=20)

    optimizer = torch.optim.SGD(model_s.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # print(model_s.parameters())
    # multi lr
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    # create recorder
    args.start_epoch = args.start_epoch or 0

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        train_acc1, train_acc5 = train(data.train_loader, model_s, criterion, optimizer, epoch, args)
        acc1, acc5 = validate(data.val_loader, model_s, criterion, args)
        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                logger.info(acc1)
                torch.save(model_s.state_dict(), 'pretrained_model/' + args.arch_s + '/' + args.set + "/T_{}_S_{}_{}.pt".format(args.arch, args.arch_s, args.set))


def load_vgg_model(model, oristate_dict):
    state_dict = model.state_dict()
    last_select_index = None  # Conv index selected in the previous layer

    mask = torch.load('/public/ly/xianyu/pretrained_model/resnet56/cifar100/cifar100_T_resnet56_S_cvgg11_bn_mask.pt')  # 要手动调整
    cnt = -1
    print(mask['layer_num'])

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            cnt += 1
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:
                select_index = torch.argsort(mask['mask'][cnt])[orifilter_num - currentfilter_num:]  # preserved filter id
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)


def load_resnet_model(model, oristate_dict, layer):
    cfg = {
        20: [3, 3, 3],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    mask = torch.load('/public/ly/xianyu/pretrained_model/cvgg16_bn/cifar100/cifar100_T_cvgg16_bn_S_resnet20_mask.pt')  # 要手动调整
    print(mask['layer_num'])
    cnt=-1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            cnt += 1
            for l in range(2):
                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight =state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    select_index = torch.argsort(mask['mask'][cnt])[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

    model.load_state_dict(state_dict)

if __name__ == "__main__":
    # setup: python finetune.py --gpu 3 --arch_s cvgg11_bn_small --set cifar10 --lr 0.01 --batch_size 256 --weight_decay 0.005 --epochs 150 --lr_decay_step 50,100  --num_classes 10 --arch cvgg16_bn
    main()