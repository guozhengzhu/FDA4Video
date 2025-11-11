import argparse
import datetime
import math
import os
import sys
import json
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import configs
from utils import MetricLogger, SmoothedValue, load_model, save_model
import models_adapter
from video_dataset import VideoDataset
from configs import DATASETS
from configs import *
import numpy as np


# 确保输出目录存在
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", type=str, required=True,
                        help='model architecture name.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size per gpu')
    parser.add_argument('--blr', type=float, default=1e-3,
                        help='base learning rate per 256 samples. actual base learning rate is linearly scaled '
                             'based on batch size.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='constant base learning rate. overrides the --blr option.')
    parser.add_argument('--weight_decay', type=float, default=5e-2,
                        help='optimizer weight decay.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs.')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='number of warmup epochs.')
    parser.add_argument('--eval_only', action='store_true',
                        help='only run evaluation.')
    parser.add_argument('--save_dir', type=str, default="",
                        help='directory to save the checkpoints in. if empty no checkpoints are saved.')
    parser.add_argument('--auto_resume', action='store_true',
                        help='automatically resume from the last checkpoint.')
    parser.add_argument('--auto_remove', action='store_true',
                        help='automatically remove old checkpoint after generating a new checkpoint.')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save checkpoint every n epochs.')
    parser.add_argument('--resume', type=str,
                        help='manually specify checkpoint to resume from. overrides --auto_resume and --pretrain.')
    parser.add_argument('--pretrain', type=str,
                        help='initialize model from the given checkpoint, discard mismatching weights and '
                             'do not load optimizer states.')
    parser.add_argument('--dataset', default="", type=str, required=True, choices=DATASETS.keys(),
                        help='name of the dataset. the dataset should be configured in config.py.')
    parser.add_argument('--mirror', action='store_true',
                        help='whether mirror augmentation (i.e., random horizontal flip) should be used during training.')
    parser.add_argument('--spatial_size', type=int, default=224,
                        help='spatial crop size.')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='number of sampled frames per video.')
    parser.add_argument('--sampling_rate', type=int, default=16,
                        help='interval between sampled frames. 0 means frames evenly covers the whole video '
                             '(i.e., with variable frame interval depending on the video length).)')
    parser.add_argument('--num_spatial_views', type=int, default=1, choices=[1, 3, 4],
                        help='number of spatial crops used for testing (only 1 and 3 supported currently).')
    parser.add_argument('--num_temporal_views', type=int, default=3,
                        help='number of temporal crops used for testing.')
    parser.add_argument('--auto_augment', type=str,
                        help='enable RandAugment of a certain configuration. see the examples in the SSv2 training scripts.')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='number of dataloader workers.')
    parser.add_argument('--resize_type', type=str, default='random_short_side_scale_jitter',
                        choices=['random_resized_crop', 'random_short_side_scale_jitter'],
                        help='spatial resize type. supported modes are "random_resized_crop" and "random_short_side_scale_jitter".'
                             'see implementation in video_dataset/transform.py for the details.')
    parser.add_argument('--scale_range', type=float, nargs=2, default=[0.08, 1.0],
                        help='range of spatial random resize. for random_resized_crop, the range limits the portion of the cropped area; '
                             'for random_short_side_scale_jitter, the range limits the target short side (as the multiple of --spatial_size).')
    parser.add_argument('--print_freq', type=int, default=10, metavar='N',
                        help='print a log message every N training steps.')
    parser.add_argument('--eval_freq', type=int, default=1, metavar='N',
                        help='evaluate on the validation set every N epochs.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use for training.')
    args = parser.parse_args()

    # 确保输出目录存在
    if args.save_dir:
        ensure_dir(args.save_dir)

    # 设置GPU设备
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"====> 使用设备: {device}")

    # 打印当前设备信息
    if torch.cuda.is_available():
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("警告: 未检测到CUDA设备，使用CPU训练")

    print("{}".format(args).replace(', ', ',\n'))

    print('创建模型')

    model = models_adapter.__dict__[args.model](
        num_classes=DATASETS[args.dataset]['NUM_CLASSES'],
        num_frames=args.num_frames  # 传递实际帧数参数
    ).to(device).train()

    # 计算可训练参数
    n_trainable_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print('可训练参数: %s, %s, %s' % (n, p.size(), p.dtype))
            n_trainable_params += p.numel()
    print('总可训练参数:', n_trainable_params, '(%.2f M)' % (n_trainable_params / 1000000))
    print("可用GPU数量:", torch.cuda.device_count())

    print('创建数据集')
    if not args.eval_only:
        dataset_train = VideoDataset(
            list_path=DATASETS[args.dataset]['TRAIN_LIST'],
            data_root=DATASETS[args.dataset]['TRAIN_ROOT'],
            random_sample=True,
            mirror=args.mirror,
            spatial_size=args.spatial_size,
            auto_augment=args.auto_augment,
            num_frames=args.num_frames,
            sampling_rate=args.sampling_rate,
            resize_type=args.resize_type,
            scale_range=args.scale_range,
        )
        print('训练数据集:', dataset_train)

    dataset_val = VideoDataset(
        list_path=DATASETS[args.dataset]['VAL_LIST'],
        data_root=DATASETS[args.dataset]['VAL_ROOT'],
        random_sample=False,
        spatial_size=args.spatial_size,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        num_spatial_views=args.num_spatial_views,
        num_temporal_views=args.num_temporal_views,
    )
    print('验证数据集:', dataset_val)

    if not args.eval_only:
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.eval_only:
        optimizer = None
        loss_scaler = None
        lr_sched = None
    else:
        # 确定学习率
        if args.lr is not None:
            print('使用绝对学习率:', args.lr)
            base_lr = args.lr
        else:
            print('使用相对学习率 (每256个样本):', args.blr)
            base_lr = args.blr * args.batch_size / 256
            print('有效学习率:', base_lr)

        # 参数分组
        params_with_decay, params_without_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if '.bias' in n:
                params_without_decay.append(p)
            else:
                params_with_decay.append(p)

        # 优化器
        optimizer = torch.optim.AdamW(
            [
                {'params': params_with_decay, 'lr': base_lr, 'weight_decay': args.weight_decay},
                {'params': params_without_decay, 'lr': base_lr, 'weight_decay': 0.}
            ],
        )
        print(optimizer)

        # 梯度缩放
        loss_scaler = torch.cuda.amp.GradScaler()

        # 学习率调度函数
        def lr_func(step):
            epoch = step / len(dataloader_train)
            if epoch < args.warmup_epochs:
                return epoch / args.warmup_epochs
            else:
                return 0.5 + 0.5 * math.cos((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * math.pi)

        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    # 评估函数
    def evaluate(log_stats=None):
        metric_logger = MetricLogger(delimiter="  ")
        header = '测试:'
        model.eval()
        for data, labels in metric_logger.log_every(dataloader_val, 100, header):
            data, labels = data.to(device), labels.to(device)
            B, V = data.size(0), data.size(1)
            data = data.flatten(0, 1)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    logits = model(data)
                scores = logits.softmax(dim=-1)
                scores = scores.view(B, V, -1).mean(dim=1)
                acc1 = (scores.topk(1, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                acc5 = (scores.topk(5, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
            metric_logger.meters['acc1'].update(acc1, n=scores.size(0))
            metric_logger.meters['acc5'].update(acc5, n=scores.size(0))

        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

        if log_stats is not None:
            log_stats.update({'val_' + k: meter.global_avg for k, meter in metric_logger.meters.items()})
            return log_stats
        return None

    start_epoch = load_model(args, model, optimizer, lr_sched, loss_scaler)

    if args.eval_only:
        evaluate()
        return

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        model.train()
        for step, (data, labels) in enumerate(metric_logger.log_every(dataloader_train, args.print_freq, header)):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(data)
                # 计算准确率
                top1 = logits.topk(1, dim=1)[1]
                acc1_all = top1 == labels.view(-1, 1)
                acc1 = acc1_all.sum(dim=-1).float().mean().item() * 100
                top5 = logits.topk(5, dim=1)[1]
                acc5 = (top5 == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                loss = F.cross_entropy(logits, labels)

            # 反向传播和优化
            loss_scaler.scale(loss).backward()

            # 确保梯度连续
            for name, param in model.named_parameters():
                if param.grad is not None and not param.grad.is_contiguous():
                    param.grad.data = param.grad.data.contiguous()

            # 1. 执行优化器更新
            loss_scaler.step(optimizer)
            # 2. 更新梯度缩放器
            loss_scaler.update()
            # 3. 只有在优化器更新后才更新学习率调度器
            # 添加条件检查确保优化器已更新
            if optimizer._step_count > 0:  # 确保优化器至少更新过一次
                lr_sched.step()
            # =====================================

            # 更新指标
            metric_logger.update(
                loss=loss.item(),
                lr=optimizer.param_groups[0]['lr'],
                acc1=acc1, acc5=acc5,
            )

            # 打印训练进度
            if step % args.print_freq == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{step}/{len(dataloader_train)}] "
                      f"Loss: {loss.item():.4f} Acc@1: {acc1:.2f}% LR: {optimizer.param_groups[0]['lr']:.2e}")

        print('平均统计:', metric_logger)
        log_stats = {'train_' + k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # 保存模型
        save_model(args, epoch, model, optimizer, lr_sched, loss_scaler)

        # 定期评估
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            log_stats = evaluate(log_stats)

        # 保存日志
        if args.save_dir is not None:
            n_total_params, n_trainable_params = 0, 0
            for n, p in model.named_parameters():
                n_total_params += p.numel()
                if p.requires_grad:
                    n_trainable_params += p.numel()

            log_stats['epoch'] = epoch
            log_stats['n_trainable_params'] = n_trainable_params
            log_stats['n_total_params'] = n_total_params

            # 确保目录存在
            ensure_dir(args.save_dir)

            log_file = os.path.join(args.save_dir, 'log.txt')
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


if __name__ == '__main__':
    # 添加安全启动机制
    try:
        main()
    except Exception as e:
        print(f"主函数发生错误: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)
