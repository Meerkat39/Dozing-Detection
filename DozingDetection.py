import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import pygame
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
import platform
from collections import deque

DOZING, NEARLY_DOZING, AWAKE = 0, 1, 2


""" 
Windowクラス 
    カメラor動画から取得した画像を表示する
"""
class MyWindow(QMainWindow):

    def __init__(self, viewer):
        super(MyWindow, self).__init__()
        self.initUI()
        self.viewer = viewer

    def initUI(self):
        """ UIの初期化 """
        menubar = self.menuBar()

        # 項目「File」の追加
        fileAction = QAction('&開く', self)
        fileAction.triggered.connect(self.setVideo)
        fileMenu = menubar.addMenu('&ファイル')
        fileMenu.addAction(fileAction)
        
        # 項目「アラーム設定」の追加
        alarm_on = QAction('&アラーム :      ON  ', self)
        alarm_off = QAction('&アラーム :      OFF  ', self)
        alarm_on.triggered.connect(self.setAlarmOn)
        alarm_off.triggered.connect(self.setAlarmOff)
        alarmMenu = menubar.addMenu('&アラーム設定')
        alarmMenu.addAction(alarm_on)
        alarmMenu.addAction(alarm_off)

        # 項目「音量設定」の追加
        volume_big = QAction('&音量 :  大 ', self)
        volume_middle = QAction('&音量 :  中 ', self)
        volume_small = QAction('&音量 :  小 ', self)
        volume_big.triggered.connect(self.setVolumeBig)
        volume_middle.triggered.connect(self.setVolumeMiddle)
        volume_small.triggered.connect(self.setVolumeSmall)
        volumeMenu = menubar.addMenu('&音量設定')
        volumeMenu.addAction(volume_big)
        volumeMenu.addAction(volume_middle)
        volumeMenu.addAction(volume_small)
    
        # 項目「直近n秒を検出」の追加
        duration_long = QAction('&直近10秒を検出(デフォルト) ', self)
        duration_neutral = QAction('&直近20秒を検出 ', self)
        duration_short = QAction('&直近30秒を検出 ', self)
        duration_long.triggered.connect(self.setDurationLong)
        duration_neutral.triggered.connect(self.setDurationNeutral)
        duration_short.triggered.connect(self.setDurationShort)
        volumeMenu = menubar.addMenu('&検出時間設定')
        volumeMenu.addAction(duration_long)
        volumeMenu.addAction(duration_neutral)
        volumeMenu.addAction(duration_short)

        # ツールバー「カメラを使用するボタン」
        cameraAct = QAction('カメラを使用する', self)
        cameraAct.triggered.connect(self.setCamera)
        self.toolbar = self.addToolBar('カメラを使用する')
        self.toolbar.addAction(cameraAct)

        # ツールバー「一時停止ボタン」
        pauseAct = QAction('一時停止', self)
        pauseAct.triggered.connect(self.pause)
        self.toolbar = self.addToolBar('一時停止')
        self.toolbar.addAction(pauseAct)

        self.resize(600, 600)                  # 600x600ピクセルにリサイズ
        self.setWindowTitle('居眠り検知ツール')  # タイトルを設定
        self.show()

    def setVideo(self):
        """ 選択されたファイルのパスを取得して、ビデオをセットする"""
        self.filepath = QFileDialog.getOpenFileName(self, caption="", directory="", filter="*.mp4")[0]
        self.viewer.setVideoCapture(False, self.filepath)            
        
    def setAlarmOn(self):
        self.viewer.setAlarm(True)
        
    def setAlarmOff(self):
        self.viewer.setAlarm(False)
        
    def setVolumeBig(self):
        self.viewer.setVolume(1)
        
    def setVolumeMiddle(self):
        self.viewer.setVolume(0.5)
        
    def setVolumeSmall(self):
        self.viewer.setVolume(0.01)

    def setInterval(self, duration_s = 10, fps = 200):
        """ 
            直近何秒の結果を使うかを変更する (fpsは200で固定とする)
            DozinDetectionで直近何フレームを使うかの値も変更する
        """
        self.viewer.setInterval(duration_s, fps)

    def setDurationLong(self):
        self.setInterval(duration_s = 10)
        
    def setDurationNeutral(self):
        self.setInterval(duration_s = 20)
        
    def setDurationShort(self):
        self.setInterval(duration_s = 30)

    def setCamera(self):
        self.viewer.setVideoCapture(True)

    def pause(self):
        self.viewer.capture.release()   


""" 
ビデオキャプチャクラス
    居眠り状態を更新して表示する
"""
class VideoCaptureView(QGraphicsView):
    repeat_interval_ms = 200  # ms 間隔で画像更新

    def __init__(self, parent=None):
        """ コンストラクタ（インスタンスが生成される時に呼び出される） """
        super(VideoCaptureView, self).__init__(parent)

        # 変数を初期化
        self.pixmap = None
        self.item = None
        self.dozingDetection = DozingDetection()
        self.is_alarm_on = True
        pygame.mixer.init()
        self.beep_sound = pygame.mixer.Sound("alarm.mp3")
        self.beep_sound.set_volume(0.5)  # 音量を初期化
        
        # VideoCaptureを初期化 (カメラからの画像取り込み)
        self.setVideoCapture(True)

        # ウィンドウの初期化
        self.scene = QGraphicsScene()   # 描画用キャンバスを作成
        self.setScene(self.scene)
        self.setVideoImage()

        # タイマー更新 (一定間隔でsetVideoImageメソッドを呼び出す)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.setVideoImage)
        self.timer.start(self.repeat_interval_ms)

    def setInterval(self, duration_s=10, fps=200):
        self.repeat_interval_ms = fps
        num_frames = duration_s * 1000 / fps
        self.dozingDetection.setNumOfFrames(num_frames)
    
    def setVideoCapture(self, isCamera = True, filepath = ""):
        """ cv2.VideoCapture関数の引数を設定する（エラー処理用） """ 

        if isCamera or filepath=="":
            # カメラを設定（内蔵カメラ(0) or 外付けカメラ1つめ(1) or その他 を設定する）
            camera_num_choices = [0, 1, -1, 2, 3, 4, 5]
            self.capture = cv2.VideoCapture(0)
            for camera_num in camera_num_choices:
                self.capture = cv2.VideoCapture(camera_num)
                # 開けたら終了
                if self.capture.isOpened() is True:
                    break
        else:
            # mp4ファイルを設定する
            self.capture = cv2.VideoCapture(filepath)
        
        # 最後にエラー処理
        if self.capture.isOpened() is False:
            raise IOError("failed in opening VideoCapture")

    def setVideoImage(self):
        """ ビデオの画像を取得して表示 """
        ret, cv_img = self.capture.read()                # ビデオキャプチャデバイスから画像を取得
        if ret == False:
            return
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # 色変換 BGR->RGB

        cv_img = self.processing(cv_img)
        height, width, dim = cv_img.shape
        bytesPerLine = dim * width                       # 1行辺りのバイト数

        self.image = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        if self.pixmap == None:                          # 初回はQPixmap, QGraphicPixmapItemインスタンスを作成
            self.pixmap = QPixmap.fromImage(self.image)
            self.item = QGraphicsPixmapItem(self.pixmap)
            self.scene.addItem(self.item)                # キャンバスに配置
        else:
            # ２回目以降はQImage, QPixmapを設定するだけ
            self.pixmap.convertFromImage(self.image)
            # 縦横の大きさの調整
            self.pixmap= self.pixmap.scaled(600, 600, Qt.KeepAspectRatio,Qt.SmoothTransformation)
            self.item.setPixmap(self.pixmap)

    def setAlarm(self, isON):
        self.is_alarm_on = isON

    def setVolume(self, value):
        self.beep_sound.set_volume(value)

    def beep(self):
        """ 異常を検知したときの処理 """
        if self.is_alarm_on == True:
            self.beep_sound.play()

    def processing(self, src):
        """ 居眠り状態を更新して表示する """
        im = src.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        # BGR表記に修正しました（おそらくputTextの引数がBGRになっている必要あり）
        red = (0, 0, 255)
        yellow = (0, 255, 255)
        green = (0, 255, 0)
        
        ret, rgb = self.capture.read()
        # 居眠り状態を取得
        state = self.dozingDetection.detectDozing(ret, rgb)
        
        # Stateとテキストと通知を更新
        self.beep_sound.stop()  # 毎回通知をストップさせる
        if state == DOZING:
            self.beep()
            color = red
            cv2.putText(rgb, "DOZING", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)
        elif state == NEARLY_DOZING:
            self.beep()
            color = yellow
            cv2.putText(rgb, "NEARLY_DOZING", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)
        elif state == AWAKE:
            color = green
            cv2.putText(rgb, "AWAKE", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)
        else:
            print("error: nothing detected")

        # 状態の色表示
        rgb = cv2.putText(rgb, 'State:', (450, im.shape[-1]+50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        rgb = cv2.putText(rgb, 'o', (555, im.shape[-1]+48), font, 1, color, 15, cv2.LINE_AA)

        # RGBからBGRに変換
        dst = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        return dst


"""
居眠り検知するクラス
    現在のまぶたの状態を計算し、居眠り状態を判定する
"""
class DozingDetection():
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt2.xml')
        self.face_parts_detector = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
        self.ear_threshold = 0.47       # EARの値から眠そうな瞼かを判定するときの閾値
        self.num_of_latest_frames = 50  # 使用する直近のフレーム数
        self.eyelid_state = deque([0] * (self.num_of_latest_frames - 1)) # 直近の瞼の状態を入れるキュー
        self.dozing_thresholds = [48, 40]      # [寝ている, 眠い] の状態を判定する閾値
    
    def setNumOfFrames(self, new_num_frames = 50):
        pre_num_frames = self.num_of_latest_frames
        self.num_of_latest_frames = new_num_frames
        self.dozing_thresholds[0] = int(self.num_of_latest_frames * 0.96) # 9.6割の時間瞼が閉じかけていたら寝ていると判定
        self.dozing_thresholds[1] = int(self.num_of_latest_frames * 0.80) # 8  割の時間瞼が閉じかけていたら眠そうと判定  
        # dequeの更新
        if pre_num_frames >= new_num_frames:       # dequeのサイズが小さくなったら  
            while len(self.eyelid_state) >= new_num_frames:
                self.eyelid_state.popleft()        # 古い情報を削る
        elif pre_num_frames < new_num_frames - 1:  # dequeのサイズが大きくなったら
            while len(self.eyelid_state) < new_num_frames - 1:
                self.eyelid_state.append(0)        # 0を追加する
    
    def calsEAR(self, eye):
        """ Eyes Aspect Ratioを計算する"""
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        eye_ear = (A + B) / (2.0 * C)
        return round(eye_ear, 3)

    def eyeMarker(self, face_mat, position):
        for i, ((x, y)) in enumerate(position):
            cv2.circle(face_mat, (x, y), 1, (255, 255, 255), -1)
            cv2.putText(face_mat, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    def isClosedEyelid(self, ret, rgb):
        """
            眠っているなら   True
            眠っていないなら False
            ただし、顔が検出できていなかったらTrueを返す
            (姿勢が悪いのは眠っているためと見なしTrue)
        """
        tick = cv2.getTickCount()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))

        # 顔検出ができていなかったら
        if len(faces) == 0:
            return True
        
        # 顔検出ができているならば
        if len(faces) > 1:
            cv2.putText(rgb, "警告: 2人以上の顔が検出されています", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1) # 警告文表示
        
        if len(faces) == 1:
            # 顔領域を取得する
            x, y, w, h = faces[0, :]
            face_gray = gray[y:(y + h), x:(x + w)]
            scale = 480 / h
            face_gray_resized = cv2.resize(face_gray, dsize=None, fx=scale, fy=scale)
            face = dlib.rectangle(0, 0, face_gray_resized.shape[1], face_gray_resized.shape[0])
            face_parts = self.face_parts_detector(face_gray_resized, face)
            face_parts = face_utils.shape_to_np(face_parts)
            # 両目のEARを計算する
            left_eye = face_parts[42:48]
            left_eye_ear = self.calsEAR(left_eye)
            right_eye = face_parts[36:42]
            right_eye_ear = self.calsEAR(right_eye)

            # region デバッグ用
            # メインウィンドウに検出した顔の四角形領域を表示する
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # メインウィンドウに左・右目のEARを表示する
            cv2.putText(rgb, "LEFT eye EAR:{} ".format(left_eye_ear), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(rgb, "RIGHT eye EAR:{} ".format(round(right_eye_ear, 3)), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            # メインウィンドウでFPSを表示する
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
            rgb = cv2.putText(rgb, "FPS:{} ".format(
            int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
            # 別のウィンドウで検出した顔領域だけをグレーで表示する
            cv2.imshow('frame_resize', face_gray_resized)
            # 別のウィンドウで左・右目の位置を6つの点で縁取る
            self.eyeMarker(face_gray_resized, left_eye)
            self.eyeMarker(face_gray_resized, right_eye)
            # endregion

            # 眠そうな瞼をしているか
            if (left_eye_ear + right_eye_ear) < self.ear_threshold:
                return True     # 眠そうな瞼
            else:
                return False    # 眠くなさそうな瞼
    
    def detectDozing(self, ret, rgb):
        if self.isClosedEyelid(ret,rgb):
            self.eyelid_state.append(1)
        else:
            self.eyelid_state.append(0)

        # 直近のフレームのうち何割が True になっているかで判定
        frame_num = len(self.eyelid_state)
        if frame_num == self.num_of_latest_frames:
            num_closed = sum(self.eyelid_state)
            self.eyelid_state.popleft()
            if num_closed > self.dozing_thresholds[0]:
                return DOZING
            elif num_closed > self.dozing_thresholds[1]:
                return NEARLY_DOZING
            else:
                return AWAKE
        else:
            cv2.putText(rgb, "警告: dequeのサイズに不備があります", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1) # 仮の警告文表示
            print("警告: dequeのサイズに不備があります")
            return AWAKE
        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    viewer = VideoCaptureView()       # VideoCaptureView ウィジエットviewを作成
    main = MyWindow(viewer)           # メインウィンドウmainを作成
    main.setCentralWidget(viewer)     # mainにviewを埋め込む
    main.show()
    app.exec_()
    viewer.capture.release()