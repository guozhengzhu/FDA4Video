# modified from https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/misc.py#L170

import os
import builtins
import datetime
from collections import defaultdict, deque
import time
import sys
import traceback

import torch

import configs

gpu_id = configs.gpu_id
device = torch.device("cuda:" + str(gpu_id))

# === 设备初始化 ===
# try:
#     # 使用 configs.gpu_id 而不是直接使用 gpu_id
#     device_str = f"cuda:{configs.gpu_id}" if torch.cuda.is_available() else "cpu"
#     device = torch.device(device_str)
#     print(f"====> utils.py: 使用设备 {device}")
# except Exception as e:
#     print(f"设备初始化失败: {e}")
#     device = torch.device("cpu")
#     print(f"回退到 CPU 设备")


def setup_for_distributed(is_master):
    """
    禁用非主进程的日志记录 - 单机环境总是启用
    """
    # 单机环境总是启用打印
    pass


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        单机环境不需要同步
        """
        pass

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """单机环境不需要同步"""
        pass

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def load_model(args, model_without_ddp, optimizer, lr_sched, loss_scaler):
    """加载模型 - 单机版本"""
    if args.resume is None and args.auto_resume:
        print('尝试从保存目录自动恢复')
        if os.path.isdir(args.save_dir):
            ckpts = [x for x in os.listdir(args.save_dir) if x.startswith('checkpoint-') and x.endswith('.pth')]
        else:
            ckpts = []
        ckpt_epochs = [int(x[len('checkpoint-'):-len('.pth')]) for x in ckpts]
        ckpt_epochs.sort()
        print(f'找到 {len(ckpt_epochs)} 个候选检查点')
        for epoch in ckpt_epochs[::-1]:
            ckpt_path = os.path.join(args.save_dir, 'checkpoint-%d.pth' % epoch)
            try:
                torch.load(ckpt_path, map_location='cpu')
            except Exception as e:
                print(f'加载检查点 {ckpt_path} 失败: {e}')
                continue
            print('找到有效检查点:', ckpt_path)
            args.resume = ckpt_path
            break
        if args.resume is None:
            print('未找到有效检查点')

    if args.resume:
        print('从检查点恢复:', args.resume)
        ckpt = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(ckpt['model'], strict=False)
        # 严格加载但仅针对需要梯度的参数
        assert len(unexpected_keys) == 0, unexpected_keys
        unexpected_keys = set(unexpected_keys)
        for n, p in model_without_ddp.named_parameters():
            if p.requires_grad:
                assert n not in missing_keys
            else:
                assert n in missing_keys, n

        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        if lr_sched is not None:
            lr_sched.load_state_dict(ckpt['lr_sched'])
        if loss_scaler is not None:
            loss_scaler.load_state_dict(ckpt['loss_scaler'])
        return ckpt['next_epoch']

    elif args.pretrain:
        print('使用预训练模型:', args.pretrain)
        ckpt = torch.load(args.pretrain, map_location='cpu')
        ckpt = ckpt['model']
        for n, p in model_without_ddp.named_parameters():
            if not p.requires_grad and n in ckpt:
                del ckpt[n]
        print(model_without_ddp.load_state_dict(ckpt, strict=False))

    return 0

def save_model(args, epoch, model_without_ddp, optimizer, lr_sched, loss_scaler):
    """保存模型 - 单机版本"""
    # 单机环境总是保存模型
    if ((epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs):
        os.makedirs(args.save_dir, exist_ok=True)
        state_dict = model_without_ddp.state_dict()
        for n, p in model_without_ddp.named_parameters():
            if not p.requires_grad:
                del state_dict[n]
        torch.save({
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'lr_sched': lr_sched.state_dict(),
            'loss_scaler': loss_scaler.state_dict(),
            'next_epoch': epoch + 1,
        }, os.path.join(args.save_dir, 'checkpoint-%d.pth' % epoch))

        if args.auto_remove:
            for ckpt in os.listdir(args.save_dir):
                try:
                    if not (ckpt.startswith('checkpoint-') and ckpt.endswith('.pth')):
                        raise ValueError()
                    ckpt_epoch = int(ckpt[len('checkpoint-'):-len('.pth')])
                except ValueError:
                    continue

                if ckpt_epoch < epoch:
                    ckpt_path = os.path.join(args.save_dir, ckpt)
                    print('移除旧检查点:', ckpt_path)
                    os.remove(ckpt_path)