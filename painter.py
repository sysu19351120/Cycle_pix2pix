import cv2
import numpy as np
import os
print("点击鼠标左键开始画图，右键使用橡皮擦工具,enter键完成绘图,esc退出")
drawing = False  # 是否开始画图
mode = True  # True：画矩形，False：画圆
start = (-1, -1)

# 鼠标的回调函数的参数格式是固定的，不要随意更改。
def mouse_event(event, x, y, flags, param):
    size = 5
    global start, drawing, mode1

    # 左键按下：开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mode1=1
        start = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        mode1 = 0
        start = (x, y)
    # 鼠标移动，画图
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode1==1:
                cv2.circle(img, (x, y), size, (0,0,0), size*2)
            else:
                cv2.circle(img, (x, y), size*2, (255,255,255),size*4)
    # 左键释放：结束画图
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode1==1:
            cv2.circle(img, (x, y), size, (0,0,0), size*2)
        else:
            cv2.circle(img, (x, y), size*2, (255,255,255), size*4)


img = np.ones((512,512,3), np.uint8)*255
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_event)
if not os.path.exists("./painting"):
    os.mkdir("./painting")
while(True):
    cv2.imshow('image', img)
    # 按下m切换模式

    if cv2.waitKey(1) == ord('m'):
        mode = not mode
    # 按ESC键退出程序
    elif cv2.waitKey(1) == 27:
        break
    elif cv2.waitKey(1)==13:
        filename = input("输入保存文件名称")
        # dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst=img
        img2=np.ones((512,512,3), np.uint8)*255
        print(dst.shape,img2.shape)
        # dst = np.concatenate([dst,img2], axis=1)
        # dst=cv2.resize(dst,(128,64))
        cv2.imwrite("painting/" + filename + '.jpg', dst)
        break