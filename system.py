from tkinter import *
import os
from tkinter import filedialog
import video
from tkinter import messagebox
import predict
import paddle
import numpy as np
import cv2
from config import get
from paddle.static import InputSpec
window = Tk()
window.title("动物影像识别系统")
window.geometry("300x200")
import end
def clicked():
    file = filedialog.askopenfilenames(initialdir=os.path.dirname(__file__))
    file=file[0]
    a = '/'
    if a in file:
        file = file.replace(a, '\\')
    lbl1.configure(text=file)
    cap = cv2.VideoCapture(file)
    frame_index = 0
    frame_count = 0
    if cap.isOpened():
        success = True
        messagebox.showinfo("Message title", "读取成功")
    else:
        success = False
        messagebox.showinfo("Message title", "读取失败")
    i=0
    while (success):
        success, frame = cap.read()
        # print "---> 正在读取第%d帧:" % frame_index, success

        if frame_index % 1== 0 and success:  # 如路径下有多个视频文件时视频最后一帧报错因此条件语句中加and success
            #resize_frame = cv2.resize(frame, (720,480), interpolation=cv2.INTER_AREA)
            cv2.imwrite(r'D:\bysj\123\{}.jpg'.format(i), frame)
            i+=1
            frame_count += 1

        frame_index += 1
    retval=cap.get(7)

def StartLearning():
    messagebox.showinfo("Message title", "已经成功运行请耐心等待")
    predict_dataset = predict.ZodiacDataset(mode='test')
    # print('测试数据集样本量：{}'.format(len(predict_dataset)))
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
    #print(result)
    # 样本映射
    LABEL_MAP = get('LABEL_MAP')
    a=end.a()
    lbl1.configure(text="测试结果："+a)

B = Button(window, text="开始计算", fg="red",command=StartLearning)
B1=Button(window,text="读取视频",fg="green",command=clicked)
lbl = Label(window,text="欢迎使用动物影像识别系统")
lbl1=Label(window,text="计算结果")

lbl.grid(column=0,row=0)
lbl1.grid(column=0,row=1)
B.grid(column=2,row=3)
B1.grid(column=2,row=4)
window.mainloop()