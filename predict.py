import paddle
import paddle.vision.transforms as T
import numpy as np
from config import get
from PIL import Image
from paddle.static import InputSpec
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



'''
predict_dataset = ZodiacDataset(mode='test')
#print('测试数据集样本量：{}'.format(len(predict_dataset)))
# 网络结构示例化
network = paddle.vision.models.resnet50(num_classes=get('num_classes'))

# 模型封装
model_2 = paddle.Model(network, inputs=[InputSpec(shape=[-1] + get('image_shape'), dtype='float32', name='image')])

# 训练好的模型加载
model_2.load(get('model_save_dir'))

# 模型配置
model_2.prepare()

# 执行预测
result = model_2.predict(predict_dataset)
print(result)

# 样本映射
LABEL_MAP = get('LABEL_MAP')

# 随机取样本展示
indexs = [2, 38, 49,56,75,92, 95,100,200,205,269,303,305,356,462,486,596,]

for idx in indexs:
    predict_label = np.argmax(result[0][idx])
    real_label = predict_dataset[idx][1]

    print('样本ID：{}, 真实标签：{}, 预测值：{}'.format(idx, LABEL_MAP[real_label], LABEL_MAP[predict_label]))
model_2.save('infer/zodiac', training=False)'''
