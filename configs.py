# === 动态设置 GPU ID ===
gpu_id = 0  # 默认值
CLIP_VIT_B16_PATH = '/home/zhangliye/zgz/fda4video/ViT-B-16.pt'
CLIP_VIT_L14_PATH = '/home/zhangliye/zgz/fda4video/ViT-L-14.pt'

# Whether cuDNN should be temporarily disable for 3D depthwise convolution.
# For some PyTorch builds the built-in 3D depthwise convolution may be much
# faster than the cuDNN implementation. You may experiment with your specific
# environment to find out the optimal option.
DWCONV3D_DISABLE_CUDNN = True
DATASETS = {
    'ssv2': dict(
        TRAIN_ROOT='/home/zhangliye/zgz/fda4video/data/SSV2/20bn-something-something-v2',
        VAL_ROOT='/home/zhangliye/zgz/fda4video/data/SSV2/20bn-something-something-v2',
        TRAIN_LIST='/home/zhangliye/zgz/fda4video/data/SSV2/train.txt',
        VAL_LIST='/home/zhangliye/zgz/fda4video/data/SSV2/val.txt',
        NUM_CLASSES=174,
    ),
    'UCF101': dict(
        TRAIN_ROOT='/home/zhangliye/zgz/fda4video/data/UCF101/train',
        VAL_ROOT='/home/zhangliye/zgz/fda4video/data/UCF101/test',
        TRAIN_LIST='/home/zhangliye/zgz/fda4video/data/UCF101/train.txt',
        VAL_LIST='/home/zhangliye/zgz/fda4video/data/UCF101/test.txt',
        NUM_CLASSES=101,
    ),
    'k400': dict(
        TRAIN_ROOT='/home/zhangliye/zgz/fda4video/data/K400',
        VAL_ROOT='/home/zhangliye/zgz/fda4video/data/K400',
        TRAIN_LIST='/home/zhangliye/zgz/fda4video/data/K400/train.txt',
        VAL_LIST='/home/zhangliye/zgz/fda4video/data/K400/val.txt',
        NUM_CLASSES=400,
    ),
    'hmdb51': dict(
        TRAIN_ROOT='/home/zhangliye/zgz/fda4video/data/hmdb51/train',
        VAL_ROOT='/home/zhangliye/zgz/fda4video/data/hmdb51/test',
        TRAIN_LIST='/home/zhangliye/zgz/fda4video/data/hmdb51/train.txt',
        VAL_LIST='/home/zhangliye/zgz/fda4video/data/hmdb51/test.txt',
        NUM_CLASSES=51,
    ),
    'Diving48': dict(
        TRAIN_ROOT='/home/zhangliye/zgz/fda4video/data/Diving48',
        VAL_ROOT='/home/zhangliye/zgz/fda4video/data/Diving48',
        TRAIN_LIST='/home/zhangliye/zgz/fda4video/data/Diving48/train.txt',
        VAL_LIST='/home/zhangliye/zgz/fda4video/data/Diving48/test.txt',
        NUM_CLASSES=48,
    ),
}