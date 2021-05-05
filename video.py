import cv2
import os
def video2frame(file):
    """
    将视频按固定间隔读取写入图片
    :param video_src_path: 视频存放路径
    :param formats:　包含的所有视频格式
    :param frame_save_path:　保存路径
    :param frame_width:　保存帧宽
    :param frame_height:　保存帧高
    :param interval:　保存帧间隔
    :return:　帧图片
    """

    '''for each_video in videos:
        # print "正在读取视频：", each_video
        print("正在读取视频：", each_video)  # 我的是Python3.6

        each_video_name = each_video[:-4]
        os.mkdir(frame_save_path + each_video_name)
        each_video_save_full_path = os.path.join(frame_save_path, each_video_name) + "/"

        each_video_full_path = os.path.join(video_src_path, each_video)'''
    cap = cv2.VideoCapture(file)
    frame_index = 0
    frame_count = 0
    if cap.isOpened():
        success = True
        global a
        a=0
    else:
        success = False
        print("读取失败!")
    i=0

    while (success):
        success, frame = cap.read()
        # print "---> 正在读取第%d帧:" % frame_index, success
        #print("---> 正在读取第%d帧:" % frame_index, success)  # 我的是Python3.6

        if frame_index % 10== 0 and success:  # 如路径下有多个视频文件时视频最后一帧报错因此条件语句中加and success
            #resize_frame = cv2.resize(frame, (720,480), interpolation=cv2.INTER_AREA)
            cv2.imwrite(r'D:\bysj\123\ox\{}.jpg'.format(i), frame)
            i+=1
            frame_count += 1

        frame_index += 1

    #cap.release()  # 这行要缩一下、原博客会报错(全局变量与局部变量)

    retval=cap.get(7)
    #print(retval)
    print(file)



