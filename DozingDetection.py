import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
import platform
DOZING, NEARLY_DOZING, AWAKE = 0, 1, 2

""" 
Windowクラス 
TODO :
・File項目
・Settings項目
・カメラの開始 / 停止
"""
class MyWindow(QMainWindow):

    def __init__(self, viewer):
        """ インスタンスが生成されたときに呼び出されるメソッド """
        super(MyWindow, self).__init__()
        self.initUI()
        self.viewer = viewer

    def initUI(self):
        """ UIの初期化 """

        """
            File -> {open} : ファイルを開く
            Setting -> {,  : ???
                        ,} : ???
            
        """
        menubar = self.menuBar()
        # 項目「File」の追加
        fileAction = QAction('&open', self)
        fileAction.triggered.connect(self.setVideo)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(fileAction)

        # 項目「Settings」の追加
        settingsAction = QAction('&', self)
        settingsAction.triggered.connect(self.close)
        settingsMenu = menubar.addMenu('&Settings')
        settingsMenu.addAction(settingsAction)

        # ツールバー「一時停止・再開ボタン」
        videoAct = QAction('一時停止・再開', self)
        videoAct.triggered.connect(self.pause_or_play)
        # （TODO: addToolBarの引数間違ってたら直す）
        self.toolbar = self.addToolBar('一時停止・再開')
        self.toolbar.addAction(videoAct)

        # ツールバー「通知」
        alarm = QAction('通知', self)
        alarm.triggered.connect(self.beep)
        self.toolbar.addAction(alarm)

        self.resize(600, 600)                  # 600x600ピクセルにリサイズ
        self.setWindowTitle('居眠り検知ツール')  # タイトルを設定
        self.show()

    def setVideo(self):
        """ 選択されたファイルのパスを取得 """
        self.filepath = QFileDialog.getOpenFileName(self, caption="", directory="", filter="*.mp4")[0]
        # TODO: viewerの中に定数を作って、それにfilepathを渡してVideoCaptureViewの引数にする

    def pause_or_play(self):
        """ 一時停止・再開ボタンが押されたときの処理 """
        # TODO: カメラのときとビデオファイルのときで場合分けする必要がありそうなので、
        # 0ならカメラ、ファイル名ならビデオ みたいな判定ができる変数が欲しい
        print('clicked: pause_or_play')
        # まだどうやって止めたり再生するのかわからない

    def beep(self):
        """ 異常を検知したときの処理（ビープ音を鳴らす.） """

        #freq : 周波数
        #dur  : 継続時間（ms）
        freq = 1400
        dur = 1000
        if platform.system() == "Windows":
            # Windowsの場合
            import winsound
            winsound.Beep(freq, dur)
        else:
            # Macの場合
            import os
            os.system('play -n synth %s sin %s' % (dur/1000, freq))

        # Warning Message box

        QMessageBox.warning(self, "警告", "居眠り検知しました.")


""" 
ビデオキャプチャクラス 
TODO :
・状態によって居眠り状態の色を更新
・リアルタイムの画像を処理するのか選択された動画の画像を処理するのかの場合分け
"""
class VideoCaptureView(QGraphicsView):
    repeat_interval = 200  # ms 間隔で画像更新

    def __init__(self, parent=None):
        """ コンストラクタ（インスタンスが生成される時に呼び出される） """
        super(VideoCaptureView, self).__init__(parent)

        # 変数を初期化
        self.pixmap = None
        self.item = None
        self.dozingDetection = DozingDetection()
        
        # VideoCapture (カメラからの画像取り込み)を初期化
        self.capture = cv2.VideoCapture(0)

        if self.capture.isOpened() is False:
            raise IOError("failed in opening VideoCapture")

        # ウィンドウの初期化
        self.scene = QGraphicsScene()   # 描画用キャンバスを作成
        self.setScene(self.scene)
        self.setVideoImage()

        # タイマー更新 (一定間隔でsetVideoImageメソッドを呼び出す)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.setVideoImage)
        self.timer.start(self.repeat_interval)

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
            self.item.setPixmap(self.pixmap)

    def processing(self, src):
        """ 画像処理 """
        im = src.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        red = (255, 0, 0)
        yellow = (255, 255, 0)
        green = (0, 255, 0)
        """ 
            状態によって色を更新(未実装) 
            DozingDetectionクラスから状態を取得し、それによってcolorを場合分け
        """
        ret, rgb = self.capture.read()  # たぶんタプルで返ってくるから分離する
        
        #rgb, faces = self.dozingDetection.detect_eyes(ret, rgb)  # facesは顔が認識できたかの結果っぽいから後で使えそう
        state = self.dozingDetection.detect_dozing(ret, rgb)
        if state == DOZING:
            # TODO: ここでなんかputTextする
            cv2.putText(rgb, "DOZING", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)
        elif state == NEARLY_DOZING:
            cv2.putText(rgb, "NEARLY_DOZING", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)
        elif state == AWAKE:
            # cv2.putText(rgb, "Sleepy eyes. Wake up!", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)
            print("AWAKE")
        else:  # エラー
            pass
            print("error")

        # TODO:
        # detect_eyes内でputTextとかimshowとか実行してるからウィンドウがわかれちゃってる説あるから
        # setVideoImageはいじらずにdetect_eyes関数の返り値とかをうまく調整してprocessing内で表示を調整できるようにするのがよさそう

        # TODO: 適切な返り値を設定した後は前回のフレームの結果とかを保持していって表示を変えるような処理をつくりたい
        color = green
        rgb = cv2.putText(rgb, 'State:', (450, im.shape[-1]+50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        rgb = cv2.putText(rgb, 'o', (555, im.shape[-1]+48), font, 1, color, 15, cv2.LINE_AA)

        dst = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        return dst


""" 
居眠り検知するクラス
TODO :
・目の状態を検出
"""
class DozingDetection():
    """元はdef __init__(self, viewer):"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt2.xml')
        self.face_parts_detector = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
        self.time_closed_eyelid = 0  # まぶたが連続で何秒とじているか

    def calc_ear(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        eye_ear = (A + B) / (2.0 * C)
        return round(eye_ear, 3)

    def eye_marker(self, face_mat, position):
        for i, ((x, y)) in enumerate(position):
            cv2.circle(face_mat, (x, y), 1, (255, 255, 255), -1)
            cv2.putText(face_mat, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    """
        眠っているなら   True
        眠っていないなら False
        ただし、顔が検出できていなかったらTrueを返す
        (姿勢が悪いのは眠っているためと見なしTrue)
    """
    def isDozing(self, ret, rgb):
        tick = cv2.getTickCount()

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))

        # 顔検出ができていなかったら
        if len(faces) != 1:
            return False

        # 顔検出ができているならば
        if len(faces) == 1:
            x, y, w, h = faces[0, :]
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face_gray = gray[y:(y + h), x:(x + w)]
            scale = 480 / h
            face_gray_resized = cv2.resize(face_gray, dsize=None, fx=scale, fy=scale)

            face = dlib.rectangle(0, 0, face_gray_resized.shape[1], face_gray_resized.shape[0])
            face_parts = self.face_parts_detector(face_gray_resized, face)
            face_parts = face_utils.shape_to_np(face_parts)

            left_eye = face_parts[42:48]
            self.eye_marker(face_gray_resized, left_eye)

            left_eye_ear = self.calc_ear(left_eye)
            cv2.putText(rgb, "LEFT eye EAR:{} ".format(left_eye_ear), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

            right_eye = face_parts[36:42]
            self.eye_marker(face_gray_resized, right_eye)

            right_eye_ear = self.calc_ear(right_eye)
            cv2.putText(rgb, "RIGHT eye EAR:{} ".format(round(right_eye_ear, 3)), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # まぶたが閉じている
            if (left_eye_ear + right_eye_ear) < 0.55:
                return True
            else:
                return False

                # この表示はあとで消す
                cv2.putText(rgb, "Sleepy eyes. Wake up!", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)
                #  TODO: ここでなにか返り値用の変数をつくるとよさそう

            cv2.imshow('frame_resize', face_gray_resized)  # これがグレーのウィンドウ

        #fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
        #rgb = cv2.putText(rgb, "FPS:{} ".format(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
        #return rgb, faces
    
    def detect_dozing(self, ret, rgb):
        if self.isDozing(ret,rgb):
            self.time_closed_eyelid += 1
        else:
            self.time_closed_eyelid = 0
        
        """
            TODO :
            閾値を設定する(マジックナンバーも避けるようにする)
            返り値を工夫する (3状態あるためTrue,Falseじゃ足らない)
            -> マクロを使う？
        """
        if self.time_closed_eyelid >= 20:
            return DOZING
        elif self.time_closed_eyelid >= 10:
            return NEARLY_DOZING
        else:
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
