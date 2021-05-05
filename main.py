import paddle
import paddle.vision.transforms as T
import numpy as np
from config import get
from PIL import Image

__all__ = ['ZodiacDataset']

# 定义图像的大小
image_shape = get('image_shape')
IMAGE_SIZE = (image_shape[1], image_shape[2])


class ZodiacDataset(paddle.io.Dataset):
    """
    十二生肖数据集类的定义
    """

    def __init__(self, mode='train'):
        """
        初始化函数
        """
        assert mode in ['train', 'test', 'valid'], 'mode is one of train, test, valid.'

        self.data = []

        with open('signs/{}.txt'.format(mode)) as f:
            for line in f.readlines():
                info = line.strip().split('\t')

                if len(info) > 0:
                    self.data.append([info[0].strip(), info[1].strip()])

        if mode == 'train':
            self.transforms = T.Compose([
                T.RandomResizedCrop(IMAGE_SIZE),  # 随机裁剪大小
                T.RandomHorizontalFlip(0.5),  # 随机水平翻转
                T.ToTensor(),  # 数据的格式转换和标准化 HWC => CHW
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像归一化
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(256),  # 图像大小修改
                T.RandomCrop(IMAGE_SIZE),  # 随机裁剪
                T.ToTensor(),  # 数据的格式转换和标准化 HWC => CHW
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像归一化
            ])

    def __getitem__(self, index):
        """
        根据索引获取单个样本
        """
        image_file, label = self.data[index]
        image = Image.open(image_file)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.transforms(image)

        return image, np.array(label, dtype='int64')

    def __len__(self):
        """
        获取样本总数
        """
        return len(self.data)
train_dataset=ZodiacDataset(mode='train')
valid_dataset=ZodiacDataset(mode='valid')
network=paddle.vision.models.resnet50(num_classes=get('num_classes'),pretrained=True)
model=paddle.Model(network)
model.summary((-1,)+tuple(get('image_shape')))
EPOCHS = get('epochs')
BATCH_SIZE = get('batch_size')

def create_optim(parameters):
    step_each_epoch = get('total_images') // get('batch_size')
    lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=get('LEARNING_RATE.params.lr'),
                                                  T_max=step_each_epoch * EPOCHS)

    return paddle.optimizer.Momentum(learning_rate=lr,
                                     parameters=parameters,
                                     weight_decay=paddle.regularizer.L2Decay(get('OPTIMIZER.regularizer.factor')))


# 模型训练配置
model.prepare(create_optim(network.parameters()),  # 优化器
              paddle.nn.CrossEntropyLoss(),        # 损失函数
              paddle.metric.Accuracy(topk=(1, 5))) # 评估指标


# 启动模型全流程训练
model.fit(train_dataset,            # 训练数据集
          valid_dataset,            # 评估数据集
          epochs=EPOCHS,            # 总的训练轮次
          batch_size=BATCH_SIZE,    # 批次计算的样本量大小
          shuffle=True,             # 是否打乱样本集
          verbose=1)
model.save(get('model_save_dir'))