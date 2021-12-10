import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np

""" 
Windowクラス 

TODO :
・File項目
・Settings項目
・カメラの開始 / 停止
"""
class MyWindow(QMainWindow):
    
    def __init__(self):
        """ インスタンスが生成されたときに呼び出されるメソッド """
        super(MyWindow, self).__init__()
        self.initUI()
        
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
        self.toolbar = self.addToolBar('一時停止・再開') # （TODO: addToolBarの引数間違ってたら直す）
        self.toolbar.addAction(videoAct)
        
        self.resize(600, 600)                  # 600x600ピクセルにリサイズ
        self.setWindowTitle('居眠り検知ツール') # タイトルを設定
        self.show()
        

    def setVideo(self):
        """ 選択されたファイルのパスを取得 """
        self.filepath = QFileDialog.getOpenFileName(self, caption="", directory="", filter="*.mp4")[0]

    
    def pause_or_play(self):
        """ 一時停止・再開ボタンが押されたときの処理 """
        # TODO: カメラのときとビデオファイルのときで場合分けする必要がありそうなので、
        # 0ならカメラ、ファイル名ならビデオ みたいな判定ができる変数が欲しい
        print('clicked: pause_or_play')
        # まだどうやって止めたり再生するのかわからない


""" 
ビデオキャプチャクラス 
TODO :
・状態によって居眠り状態の色を更新
・リアルタイムの画像を処理するのか選択された動画の画像を処理するのかの場合分け
"""
class VideoCaptureView(QGraphicsView):
    repeat_interval = 200 # ms 間隔で画像更新

    def __init__(self, parent = None):
        """ コンストラクタ（インスタンスが生成される時に呼び出される） """
        super(VideoCaptureView, self).__init__(parent)
        
        # 変数を初期化
        self.pixmap = None
        self.item = None
        
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
        cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)  # 色変換 BGR->RGB
        
        cv_img = self.processing(cv_img)

        height, width, dim = cv_img.shape
        bytesPerLine = dim * width                       # 1行辺りのバイト数
        
        self.image = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        if self.pixmap == None:                          # 初回はQPixmap, QGraphicPixmapItemインスタンスを作成
            self.pixmap = QPixmap.fromImage(self.image)
            self.item = QGraphicsPixmapItem(self.pixmap)
            self.scene.addItem(self.item)                # キャンバスに配置
        else:
            self.pixmap.convertFromImage(self.image)     # ２回目以降はQImage, QPixmapを設定するだけ
            self.item.setPixmap(self.pixmap)
    
    def processing(self, src):
        """ 画像処理 """
        im = src.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        im = cv2.putText(im, 'State:', (450,im.shape[-1]+50), font, 1, (0,0,0), 2, cv2.LINE_AA)
        red = (255,0,0)
        yellow = (255,255,0)
        green = (0,255,0)
        """ 
            状態によって色を更新(未実装) 
            DozingDetectionクラスから状態を取得し、それによってcolorを場合分け
        """
        color = green
        im = cv2.putText(im, 'o', (555,im.shape[-1]+48), font, 1, color, 15, cv2.LINE_AA)
      
        dst = im
        
        return dst
    
""" 
居眠り検知するクラス
TODO :
・目の状態を検出
"""
class DozingDetection():
    hoge = 0



if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    
    main = MyWindow()                 # メインウィンドウmainを作成
    viewer = VideoCaptureView()       # VideoCaptureView ウィジエットviewを作成
    main.setCentralWidget(viewer)     # mainにviewを埋め込む
    main.show()
    
    app.exec_()
    
    viewer.capture.release()
