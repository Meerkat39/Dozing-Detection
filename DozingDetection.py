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

        # ツールバー「カメラをセットするボタン」
        cameraAct = QAction('カメラを使用する', self)
        cameraAct.triggered.connect(self.setCamera)
        # （TODO: addToolBarの引数間違ってたら直す）
        self.toolbar = self.addToolBar('カメラを使用する')
        self.toolbar.addAction(cameraAct)

        # ツールバー「一時停止ボタン」
        pauseAct = QAction('一時停止', self)
        pauseAct.triggered.connect(self.pause)
        # （TODO: addToolBarの引数間違ってたら直す）
        self.toolbar = self.addToolBar('一時停止')
        self.toolbar.addAction(pauseAct)

        # ツールバー「通知」
        alarmAct = QAction('通知', self)
        alarmAct.triggered.connect(self.beep)
        self.toolbar.addAction(alarmAct)

        self.resize(600, 600)                  # 600x600ピクセルにリサイズ
        self.setWindowTitle('居眠り検知ツール')  # タイトルを設定
        self.show()

    def setVideo(self):
        """ 選択されたファイルのパスを取得して、 """
        self.filepath = QFileDialog.getOpenFileName(self, caption="", directory="", filter="*.mp4")[0]
        self.viewer.setVideoCapture(False, self.filepath)

    def setCamera(self):
        """ カメラに切り替える """
        self.viewer.setVideoCapture(True)

    def pause(self):
        """ 一時停止ボタンが押されたときの処理 """
        self.viewer.capture.release()

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
        
        # VideoCaptureを初期化 (カメラからの画像取り込み)
        self.setVideoCapture(True)

        # ウィンドウの初期化
        self.scene = QGraphicsScene()   # 描画用キャンバスを作成
        self.setScene(self.scene)
        self.setVideoImage()

        # タイマー更新 (一定間隔でsetVideoImageメソッドを呼び出す)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.setVideoImage)
        self.timer.start(self.repeat_interval)
    
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

    def processing(self, src):
        """ 画像処理 """
        im = src.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        # BGR表記に修正しました（おそらくputTextの引数がBGRになっている必要あり）
        red = (0, 0, 255)
        yellow = (0, 255, 255)
        green = (0, 255, 0)
        
        ret, rgb = self.capture.read()
        # 居眠り検出関数実行
        state = self.dozingDetection.detect_dozing(ret, rgb)
        # 状態表示
        if state == DOZING:
            color = red
            cv2.putText(rgb, "DOZING", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)
        elif state == NEARLY_DOZING:
            color = yellow
            cv2.putText(rgb, "NEARLY_DOZING", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)
        elif state == AWAKE:
            color = green
            cv2.putText(rgb, "AWAKE", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)
            pass
        else:
            print("error: nothing detected (?)")

        # 状態の色表示
        rgb = cv2.putText(rgb, 'State:', (450, im.shape[-1]+50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        rgb = cv2.putText(rgb, 'o', (555, im.shape[-1]+48), font, 1, color, 15, cv2.LINE_AA)

        # RGBからBGRに変換
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

    def calsEAR(self, eye):
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
    def isClosedEyelid(self, ret, rgb):
        tick = cv2.getTickCount()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))

        # 顔検出ができていなかったら
        if len(faces) == 0:
            return True

        # 顔検出ができているならば
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

            # TODO:
            # この辺のデバッグ用のputTextはあとで削除する
            # -> 削除するのではなく、コメント文にする
            # (cv2.rectangleは顔検出ができてるかわかるものなので常に表示してもいいかも)
            # 豆知識:
            # regionは折りたたむことができる
            # Visual Studio Codeでは、
            # 選択した範囲を Ctrl + K -> Ctrl + C でコメント文にする
            # 選択した範囲を Ctrl + K -> Ctrl + U でコメント文を解除する
            # ことができます！

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
            self.eye_marker(face_gray_resized, left_eye)
            self.eye_marker(face_gray_resized, right_eye)
            # endregion

            # 眠そうな瞼をしている
            if (left_eye_ear + right_eye_ear) < 0.55:
                return True
            else:
                return False
    
    def detect_dozing(self, ret, rgb):
        if self.isClosedEyelid(ret,rgb):
            self.time_closed_eyelid += 1
        else:
            self.time_closed_eyelid = 0
        
        """
            TODO :
            閾値を設定する(マジックナンバーも避けるようにする)
            返り値を工夫する (3状態あるためTrue,Falseじゃ足らない)
            -> マクロを使う？

            TODO :
            瞼を開き続けたり閉じ続けたりして試してみたら、
            どうやら正しい状態検知にはなっていなさそうだったのでここは修正必須っぽい。
            目の状態を定性的に考えてもうちょと細かい処理を加えたい
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
