''' PyQt5 依赖 '''
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QDateTime, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.uic import loadUi
from threading import *
import sys, cv2

''' insightface 依赖 '''
import numpy as np
import insightface
assert insightface.__version__>='0.3'
from insightface.app import FaceAnalysis

''' GUI '''
class MyMainWindow(QMainWindow):

    signal = pyqtSignal(float) # 弹窗信号

    def __init__(self):
        super(MyMainWindow, self).__init__()
    
        ''' 初始化前台 '''
        self.setWindowIcon(QIcon("./pics/logo.jpg"))
        loadUi("./client.ui", self)
        self.setFixedSize(self.width(), self.height()) # 固定窗口大小

        ''' 初始化后台 '''
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        ''' 动态日期时间 '''
        timer = QTimer(self)
        timer.timeout.connect(self.showTime)
        timer.start()

        ''' 参数初始化 '''
        self.id_photo = None # 证件照
        self.monitor_photo = None # 监控照
        self.camera = False # 摄像头开启状态
        self.take_photo = False # 抓拍状态
        self.pb.reset() # 进度条重置
        self.pb.setOrientation(Qt.Horizontal) # 水平进度

        ''' 菜单栏 '''
        self.openID.triggered.connect(self.openIDFunc)
        self.resetID.triggered.connect(self.resetIDFunc)
        self.openCamera.triggered.connect(self.openCameraWrapper)
        self.closeCamera.triggered.connect(self.closeCameraFunc)
        self.takePhoto.triggered.connect(self.takePhotoFunc)
        self.savePhoto.triggered.connect(self.savePhotoFunc)
        self.faceRecog.triggered.connect(self.faceRecogWrapper)
        self.helpBook.triggered.connect(self.helpBookFunc)
        self.aboutMe.triggered.connect(self.aboutMeFunc)

        ''' 人脸验证信号 '''
        self.signal.connect(self.messageBox)

    ''' 人脸验证结果弹窗 '''
    def messageBox(self, sim):
        self.pbStopFunc() # 停止进度条
        if sim == -1: QMessageBox.information(self, "提醒", "证件照或监控照缺失！", QMessageBox.Ok)
        elif sim > 0.24: QMessageBox.information(self, "人脸1：1验证", "通过！", QMessageBox.Ok) 
        else: QMessageBox.information(self, "人脸1：1验证", "不通过！", QMessageBox.Ok)       

    ''' 加载日期时间 '''
    def showTime(self):
        datetime = QDateTime.currentDateTime()
        text = datetime.toString()
        self.date.setText(text)

    ''' 上传证件照 '''
    def openIDFunc(self):
        img_path, _ = QFileDialog.getOpenFileName(self, "上传证件照", "./", "Image Files (*.jpg *.png)")
        if img_path == "": return # 取消上传
        self.id_photo = cv2.imread(img_path) # 缓存证件照
        self.IdArea.setPixmap(QPixmap(img_path)) # 加载证件照
        self.IdArea.setScaledContents(True) # 自适应调整分辨率

    ''' 重置证件照 '''
    def resetIDFunc(self):
        self.IdArea.setPixmap(QPixmap(""))
        self.id_photo = None

    ''' 打开摄像头 '''
    def openCameraWrapper(self):
        self.camera = True # 打开摄像头
        thread = Thread(target=self.openCameraFunc) # 放入线程避免卡机
        thread.start() # 开始推流

    def openCameraFunc(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 450) # 设置宽度
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 450) # 设置高度
        cap.set(cv2.CAP_PROP_FPS, 30) # 设置帧率
        while cap.isOpened(): # 摄像推流
            if not self.camera: break
            if self.take_photo:
                self.monitor_photo = frame_rgb # 缓存监控照片
                self.subMonitorArea.setPixmap(QPixmap(img))
                self.subMonitorArea.setScaledContents(True) # 自适应调整分辨率
                self.take_photo = False # 停止抓拍
            ret, frame = cap.read() # frame：当前帧图像
            if not ret: continue # ret：是否拍到图像
            frame_rgb = cv2.flip(frame, 1) # 镜像翻转
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            img = QImage(frame_bgr.data, frame_bgr.shape[1], frame_bgr.shape[0], QImage.Format_RGB888)
            self.MonitorArea.setPixmap(QPixmap(img)) # 加载当前帧图像
        if not self.camera: 
            self.MonitorArea.setPixmap(QPixmap("")) # 清空监控区域
            self.subMonitorArea.setPixmap(QPixmap("")) # 清空监控子窗区域

    ''' 关闭摄像头 '''
    def closeCameraFunc(self):
        self.camera = False
        
    ''' 抓拍 '''
    def takePhotoFunc(self):
        self.take_photo = True

    ''' 保存 '''
    def savePhotoFunc(self):
        if self.monitor_photo is not None:
            cv2.imwrite("./pics/monitor_photo.jpg", self.monitor_photo)
            QMessageBox.information(self, "提醒", "保存成功！", QMessageBox.Ok)
        else: QMessageBox.information(self, "提醒", "保存失败！未进行抓拍！", QMessageBox.Ok)    

    ''' 人脸验证 '''
    def faceRecogWrapper(self):
        thread = Thread(target=self.faceRecogFunc)
        thread.start()

    def faceRecogFunc(self):
        self.pbRunFunc() # 开启进度条
        if self.id_photo is None or self.monitor_photo is None:
            self.signal.emit(-1) # 发射弹窗信号
            return
        face1, face2 = self.app.get(self.id_photo), self.app.get(self.monitor_photo)
        feat1 = np.array(face1[0].normed_embedding, dtype=np.float32)
        feat2 = np.array(face2[0].normed_embedding, dtype=np.float32)
        sim = np.dot(feat1, feat2.T)
        print("sim:", sim) # 控制台打印
        self.signal.emit(sim) # 发射弹窗信号
        
    ''' 开启进度条 '''
    def pbRunFunc(self):
        self.pb.setMinimum(0)
        self.pb.setMaximum(0)

    ''' 停止进度条 '''
    def pbStopFunc(self):
        self.pb.setMinimum(0)
        self.pb.setMaximum(1)

    ''' 使用说明 '''
    def helpBookFunc(self):
        text = "《系统使用说明》\n" + "介绍：本系统采用人脸1：1验证，使用者需" + \
        "要事先提供证件照，然后在具有摄像头的设备上采集监控照，进行人脸识别即可。\n" + \
        "1. 菜单栏 -> 证件照：上传以及重置证件照。\n" + \
        "2. 菜单栏 -> 监控照：打开以及关闭摄像头，抓拍以及保存监控照。\n" + \
        "3. 菜单栏 -> 人脸识别：进行人脸1：1验证。\n" + \
        "4. 菜单栏 -> 帮助：使用说明以及关于作者。\n"
        QMessageBox.information(self, "使用说明", text, QMessageBox.Ok)

    ''' 关于作者 '''
    def aboutMeFunc(self):
        text = "哈喽~我是阿正，一个热爱编程、音乐、游戏、美食的boy。" + \
        "目前最大的愿望是能有自己的一个独立的房子，在里面养一些猫猫狗狗。" + \
        "猫就是蓝猫，狗选柴犬吧。然后组装一台炫酷的主机，硬件可以不到位，氛围灯一定要够造势。" + \
        "工作以后再买一辆代步车吧，目前倾向于Model Y，满满的科技感。" + \
        "最后就是旅行了，趁年轻到处跑跑，记录一下祖国的大好河山~"    
        QMessageBox.information(self, "关于作者", text, QMessageBox.Ok)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MyMainWindow()
    win.show()
    sys.exit(app.exec_())
