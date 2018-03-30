#!/usr/bin/python3
# -*- coding: utf-8 -*-

import datetime
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pyedflib

import random  #потом удалить

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.form_widget = FormWidget(self)
        self.setCentralWidget(self.form_widget)
        self.channels = []
        self.initUI()

    def initUI(self):
        openFile = QAction(QIcon('icons/open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.OpenFileDialog)

        self.spectralAnalysis = QAction(QIcon('icons/open.png'), 'Spectral', self)
        # spectralAnalysis.setShortcut('Ctrl+L')
        self.spectralAnalysis.setStatusTip('Spectral Analysis')
        self.spectralAnalysis.triggered.connect(self.spectralAnalysisDialog)
        self.spectralAnalysis.setEnabled(False)

        self.entropyAnalysis = QAction(QIcon('icons/open.png'), 'Entropy', self)
        # entropyAnalysis.setShortcut('Ctrl+O')
        self.entropyAnalysis.setStatusTip('Entropy Analysis')
        self.entropyAnalysis.triggered.connect(self.entropyAnalysisDialog)
        self.entropyAnalysis.setEnabled(False)

        self.startTest = QAction(QIcon('icons/open.png'), 'Start Test', self)
        # startTest.setShortcut('Ctrl+T')
        self.startTest.setStatusTip('Start Test')
        self.startTest.triggered.connect(self.startRaschTestDialog)
        self.startTest.setEnabled(True)

        self.stopTest = QAction(QIcon('icons/open.png'), 'Stop Test', self)
        # stopTest.setShortcut('Ctrl+P')
        self.stopTest.setStatusTip('Stop Test')
        self.stopTest.triggered.connect(self.stopRaschTestDialog)
        self.stopTest.setEnabled(False)

        #UnitTest для прогонки результатов
        self.unitTest = QAction(QIcon('icons/open.png'), 'Unit Test', self)
        # startUnitTest.setShortcut('Ctrl+U')
        self.unitTest.setStatusTip('Unit Test')
        self.unitTest.triggered.connect(self.unitTestFunc)
        self.unitTest.setEnabled(True)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        analysisMenu = menubar.addMenu('&Analysis')
        analysisMenu.addAction(self.spectralAnalysis)
        analysisMenu.addAction(self.entropyAnalysis)
        testMenu = menubar.addMenu('&Test')
        testMenu.addAction(self.startTest)
        testMenu.addAction(self.stopTest)
        testMenu.addAction(self.unitTest)

        self.statusbar = self.statusBar()
        self.showMaximized()
        self.setWindowTitle('EEG Analyzer')
        self.show()

    def OpenFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '')[0]
        self.eegFile = pyedflib.EdfReader(fname)
        self.ChannelChooseForm = ChannelChooseForm(self, self.eegFile)


    def spectralAnalysisDialog(self):
        self.form_widget.analyzeplot.plotSpectralAnalysisData(channels=self.channels,
                                                              StartPoint = int(self.form_widget.BottomRightWidgets['AnalysisFrameStartEdit'].text()),
                                                              EndPoint = int(self.form_widget.BottomRightWidgets['AnalysisFrameEndEdit'].text()),
                                                              file = self.eegFile)

    def entropyAnalysisDialog(self):
        pass

    def startRaschTestDialog(self):
        self.RaschTest_window = RaschTestWindow(self)
        self.startTest.setEnabled(False)
        self.stopTest.setEnabled(True)

    def stopRaschTestDialog(self):
        testNum, ok = QInputDialog.getText(self, 'Остановка теста',
                                        'Вы набрали '+str(self.RaschTest_window.checkResult())+' балов. Если вы готовы закончить тест введите свой номер:')
        self.RaschTest_window.saveResult(testNum)

    def unitTestFunc(self):
        self.startRaschTestDialog()
        self.RaschTest_window.answer = {1:'А', 2:'А', 3:'А', 4:'А', 5:'А',
                            6:'А', 7:'А', 8:'А', 9:'А', 10:'А',
                            11:'А', 12:'А', 13:'А', 14:'А', 15:'А',
                            16:'А', 17:'А', 18:'А', 19:'А', 20:'А',
                            21:'А&Б&В&Г', 22:'А&Б&В&Г', 23:'А&Б&В&Г', 24:'А&Б&В&Г',
                            25:'11&22', 26:'11&22',
                            27:'11', 28:'11', 29:'11', 30:'11',
                            31:'11&22&33&44',
                            32:'11',
                            33:'11'}
        self.stopRaschTestDialog()


class ChannelChooseForm(QWidget):

    def __init__(self, parent, eegFile):
        super().__init__(parent, Qt.Window)
        self.initUI(eegFile)
        self.parent = parent
        self.show()

    def initUI(self, eegFile):
        availablescreen = QDesktopWidget().availableGeometry()  # define screen size
        self.setWindowTitle("Выберите каналы данных")
        self.move(30, 30)
        self.ChannelCheckboxes={}
        i = 0
        for SignalChannel in eegFile.getSignalLabels():
            self.ChannelCheckboxes[i] = QCheckBox(eegFile.getSignalLabels()[i], self)
            self.ChannelCheckboxes[i].move(30+i//10*100, 30+i%10*30)
            i+=1

        self.PlotButton = QPushButton('Plot', self)
        self.PlotButton.clicked.connect(self.PlotDataFromChoosenChanels)
        self.PlotButton.move(20,320)
        self.CloseButton = QPushButton('Close', self)
        self.CloseButton.clicked.connect(self.close)
        self.CloseButton.move(120, 320)
        self.adjustSize()

    def PlotDataFromChoosenChanels(self):
        channels = []
        for key in self.ChannelCheckboxes.keys():
            if self.ChannelCheckboxes[key].isChecked():
                channels += [key]
        self.parent.channels = channels
        self.parent.form_widget.eegplot.plotEEGData(channels = channels, file = self.parent.eegFile)
        self.parent.spectralAnalysis.setEnabled(True)
        self.parent.entropyAnalysis.setEnabled(True)


class FormWidget(QWidget):

    def __init__(self, parent):
        super(FormWidget, self).__init__(parent)
        self.initUI()

    def initUI(self):
        availablescreen = QDesktopWidget().availableGeometry() #define screen size

        self.hboxtop = QHBoxLayout()

        self.EEGFrame = QFrame(self)
        self.EEGFrame.setFrameShape(QFrame.StyledPanel)

        self.EEGSlider = QSlider(Qt.Horizontal, self)
        self.EEGSlider.setFocusPolicy(Qt.NoFocus)
        # self.EEGSlider.valueChanged[int].connect(self.changeValue)

        self.EEGLayout = QVBoxLayout()
        self.EEGLayout.addWidget(self.EEGFrame)
        self.EEGLayout.addWidget(self.EEGSlider)
        self.EEGLayout.setSpacing(0)
        self.EEGLayout.setContentsMargins(0,0,0,0)

        self.EEGWidget = QWidget()
        self.EEGWidget.setGeometry(availablescreen.top(),
                                  availablescreen.left(),
                                  availablescreen.width() * 2 / 3 + 7,
                                  availablescreen.height() * 0.8)
        self.EEGWidget.setLayout(self.EEGLayout)

        self.AnalysisFrame = QFrame(self)
        self.AnalysisFrame.setFrameShape(QFrame.StyledPanel)
        self.AnalysisFrame.setGeometry(availablescreen.width() * 2 / 3 + 7,
                                       availablescreen.left(),
                                       availablescreen.width() * 1 / 3,
                                       availablescreen.height() * 0.8)


        self.topSplitter = QSplitter(Qt.Horizontal)
        self.topSplitter.addWidget(self.EEGWidget)
        self.topSplitter.addWidget(self.AnalysisFrame)

        self.hboxtop.addWidget(self.topSplitter)


        self.hboxbottom = QHBoxLayout()

        self.bottomleft = QFrame(self)
        self.bottomleft.setFrameShape(QFrame.StyledPanel)


        self.bottomcenter = QFrame(self)
        self.bottomcenter.setFrameShape(QFrame.StyledPanel)

        self.bottomright = QFrame(self)
        self.BottomrRightGrid = QGridLayout()
        self.bottomright.setFrameShape(QFrame.StyledPanel)
        self.BottomRightWidgetsGridNames = ['AnalysisFrameStartLabel','AnalysisFrameFinishLabel',
                                        'AnalysisFrameStartEdit','AnalysisFrameEndEdit'
                                        ]
        self.BottomRightGridWidth = 2
        self.BottomrRightGridHeight = 2
        self.BottomRightWidgetGridPositions = [(i, j) for i in range(self.BottomRightGridWidth) for j in range(self.BottomrRightGridHeight)]
        self.BottomRightWidgets = {}
        for position, name in zip(self.BottomRightWidgetGridPositions, self.BottomRightWidgetsGridNames):
            if name[-5:] == "Label":
                label = QLabel(name)
                self.BottomRightWidgets[name] = label
                self.BottomrRightGrid.addWidget(label, *position)
                print("label find");
            elif name[-4:] == "Edit":
                lineedit = QLineEdit(name)
                self.BottomRightWidgets[name] = lineedit
                self.BottomRightWidgets[name].setValidator(QIntValidator(0, 400000)) #Хардкод, надо переписать
                self.BottomrRightGrid.addWidget(lineedit, *position)
                print("edit find");
            else:
                print("noone find");
                continue

        self.bottomright.setGeometry(availablescreen.left(),
                                     availablescreen.top(),
                                       availablescreen.width() * 1 / 3,
                                       availablescreen.height() * 0.2)
        self.bottomright.setLayout(self.BottomrRightGrid)


        self.hboxbottom.addWidget(self.bottomleft,1)
        self.hboxbottom.addWidget(self.bottomcenter,1)
        self.hboxbottom.addWidget(self.bottomright,1)
        self.hboxbottom.setSpacing(7)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hboxtop,4)
        self.vbox.addLayout(self.hboxbottom,1)
        self.vbox.setSpacing(7)

        self.setLayout(self.vbox)

        self.plottingEEG()
        self.plottingAnalyzePlot()

    def plottingEEG(self):

        self.eegplot = PlotCanvas(self.EEGFrame, 100, 16.15, 100)

    def plottingAnalyzePlot(self):

        self.analyzeplot = PlotCanvas(self.AnalysisFrame, 12.5, 16.56, 100)

    def changeScale(self):
        pass


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=1, height=1, dpi=100):
        self.fig = mpl.figure.Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.fig.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001, wspace = 0.001, hspace = 0.001)
        self.setParent(parent)
        self.band_color = ['red', 'blue', 'green', 'yellow']

    # Clear Axis from unnecessary labels and spines
    def clear_axis(self, ax):
        ax.axes.get_yaxis().set_ticks([])  # clear y axis
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.axes.tick_params(axis='x', labelsize=8)

    def add_grid(self, ax, f, channel):
        # minor_ticks = np.arange(-500, 500, 20)
        # ax.axes.set_yticks(minor_ticks, minor=True)
        ax.axes.grid(color='blue', linestyle='--', axis='x', which='both')
        ax.axes.grid(color='blue', linestyle='--', axis='y', which='major')
        ax.axes.grid(color='blue', linestyle=':', axis='y', which='minor', alpha=0.4)
        ax.yaxis.grid(True, which='both')
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: x / f.getSampleFrequency(channel)))

    def plotEEGData(self, channels, file):
        i = 0
        for channel in channels:
            i += 1
            signal = file.readSignal(channel - 1)
            ax = self.fig.add_subplot(len(channels), 1, i )
            self.clear_axis(ax)
            self.add_grid(ax, file, channel)
            ax.plot(signal, linewidth=0.2, color='black')
            self.draw()

    def plotSpectralAnalysisData(self, channels, StartPoint, EndPoint, file):
        # Compute and plot spectral analyzis for signal (hardcoded for that time, need to be parametrized)
        i = 0
        for channel in channels:
            i += 1
            signal = file.readSignal(channel - 1)
            ax_spectr = self.fig.add_subplot(len(channels), 1, i)
            ax_spectr.set_xscale('log')
            Power, PowerRatio, PowerFreq = self.spectral_analyzis(signal, file.getSampleFrequency(channel-1), [0.5, 4, 8, 12, 35], StartPoint, EndPoint, 10)
            for color_iter in range(4):
                ax_spectr.fill_between(
                    PowerFreq[int(len(PowerFreq) / 4) * color_iter:int(len(PowerFreq) / 4) * (color_iter + 1)],
                    PowerRatio[int(len(PowerRatio) / 4) * color_iter:int(len(PowerRatio) / 4) * (color_iter + 1)],
                    alpha=0.3, linewidth=0.4, color=self.band_color[color_iter])
            self.draw()

    def spectral_analyzis(self, Signal, SignalFreq, Band, EpochStart=0, EpochStop=None, DFreq=1):
        '''
        :param Signal: list, 1-D real signal
        :param EpochStart: integer,
        :param EpochStop: integer,
        :param SignalFreq: integer, Signal physical frequency
        :param Band: list, real frequencies (in Hz) of bins  Each element of Band is a physical frequency and shall not exceed the Nyquist frequency, i.e., half of sampling frequency.
        :param DFreq: integer, number of equal segments in each band
        :return: Power: list, 2-D power in each Band divided on equal DFreq segments
        :return: PowerRatio: spectral power in each segment normalized by total power in ALL frequency bins.
        :return: PowerFreq: Frequencies in wich Power computed
        '''
        SignalSection = Signal[EpochStart:EpochStop]
        fftSignal = abs(np.fft.fft(SignalSection))
        Power = np.zeros((len(Band) - 1) * DFreq)
        PowerFreq = np.zeros((len(Band) - 1) * DFreq)

        for BandIndex in range(0, len(Band) - 1):
            Freq = float(Band[BandIndex])
            NextFreq = float(Band[BandIndex + 1])
            for FreqDiff in range(1, DFreq + 1):
                FreqD = Freq + (NextFreq - Freq) * ((FreqDiff - 1) / DFreq)
                NextFreqD = Freq + (NextFreq - Freq) * (FreqDiff / DFreq)
                Power[BandIndex * DFreq + FreqDiff - 1] = sum(fftSignal[int(
                    np.floor(FreqD / SignalFreq * len(SignalSection))): int(
                    np.floor(NextFreqD / SignalFreq * len(SignalSection)))])
                PowerFreq[BandIndex * DFreq + FreqDiff - 1] = FreqD
        PowerRatio = Power / sum(Power)
        return Power, PowerRatio, PowerFreq









#############################################################################################
############################             ZNO Rash Test            ###########################
#############################################################################################

class RaschTestWindow(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.qnum = 1
        self.answersConstDict = {1:'А', 2:'А', 3:'А', 4:'А', 5:'А',
                            6:'А', 7:'А', 8:'А', 9:'А', 10:'А',
                            11:'А', 12:'А', 13:'А', 14:'А', 15:'А',
                            16:'А', 17:'А', 18:'А', 19:'А', 20:'А',
                            21:'А&Б&В&Г', 22:'А&Б&В&Г', 23:'А&Б&В&Г', 24:'А&Б&В&Г',
                            25:'11&22', 26:'11&22',
                            27:'11', 28:'11', 29:'11', 30:'11',
                            31:'11&22&33&44',
                            32:'11',
                            33:'11'}
        self.answer = {}
        self.initUI()


    def initUI(self):
        availablescreen = QDesktopWidget().availableGeometry()  # define screen size

        self.setWindowTitle("Тест ЗНО")

        self.questionLabel = QLabel(self)
        self.questionImage = QLabel(self)

        self.move(30, 30)

        self.answersLabel = QLabel(self)
        self.answers= QFrame(self)
        self.answer1 = QComboBox(self)
        self.answer1.addItems(['А', 'Б', 'В', 'Г', 'Д'])
        self.answer2 = QComboBox(self)
        self.answer2.addItems(['А', 'Б', 'В', 'Г', 'Д'])
        self.answer2.setVisible(False)
        self.answer3 = QComboBox(self)
        self.answer3.addItems(['А', 'Б', 'В', 'Г', 'Д'])
        self.answer3.setVisible(False)
        self.answer4 = QComboBox(self)
        self.answer4.addItems(['А', 'Б', 'В', 'Г', 'Д'])
        self.answer4.setVisible(False)
        self.answerEdit1 = QLineEdit(self)
        self.answerEdit1.setVisible(False)
        self.answerEdit2 = QLineEdit(self)
        self.answerEdit2.setVisible(False)
        self.answerEdit3 = QLineEdit(self)
        self.answerEdit3.setVisible(False)
        self.answerEdit4 = QLineEdit(self)
        self.answerEdit4.setVisible(False)
        hbox = QHBoxLayout()
        hbox.addWidget(self.answer1)
        hbox.addWidget(self.answer2)
        hbox.addWidget(self.answer3)
        hbox.addWidget(self.answer4)
        hbox.addWidget(self.answerEdit1)
        hbox.addWidget(self.answerEdit2)
        hbox.addWidget(self.answerEdit3)
        hbox.addWidget(self.answerEdit4)
        self.answers.setLayout(hbox)

        self.buttons = QFrame(self)
        self.nextButton = QPushButton('Следующий вопрос', self)
        self.nextButton.clicked.connect(lambda: self.showQuestion(self.qnum))

        self.qNumber = QLineEdit(str(self.qnum))
        self.qNumber.setObjectName('qNumber')
        self.qNumber.setValidator(QIntValidator(1, 32))
        self.qNumber.returnPressed.connect(lambda: self.showQuestion(self.qnum))

        self.previousButton = QPushButton('Предидущий вопрос', self)
        self.previousButton.clicked.connect(lambda: self.showQuestion(self.qnum))
        self.previousButton.setEnabled(False)
        bbox = QHBoxLayout()
        bbox.addWidget(self.previousButton)
        bbox.addStretch(1)
        bbox.addWidget(self.qNumber)
        bbox.addStretch(1)
        bbox.addWidget(self.nextButton)
        self.buttons.setLayout(bbox)

        self.statusLabel = QLabel(self)

        vbox = QVBoxLayout()
        vbox.addWidget(self.questionLabel)
        vbox.addWidget(self.questionImage)
        vbox.addStretch(1)
        vbox.addWidget(self.answersLabel)
        vbox.addWidget(self.answers)
        vbox.addWidget(self.buttons)
        vbox.addWidget(self.statusLabel)
        self.setLayout(vbox)

        self.questionLabel.setText("Вопрос №" + str(self.qnum))
        self.questionLabel.adjustSize()
        self.questionImage.setPixmap(QPixmap('resourses/zno/' + str(self.qnum) + '.png'))
        self.questionImage.adjustSize()
        self.statusLabel.setText(str(self.answer))

        self.adjustSize()

        self.show()

    def showQuestion(self, qnum = None):
        # Сохраняем текущий ответ на вопрос в словарь
        if self.qnum in [21, 22, 23, 24]:
            self.answer[self.qnum]= self.answer1.currentText()+self.answer2.currentText()+self.answer3.currentText()+self.answer4.currentText()
        elif self.qnum in [25, 26]:
            self.answer[self.qnum] = self.answerEdit1.text()+'&'+self.answerEdit2.text()
        elif self.qnum in [27, 28, 29, 30]:
            self.answer[self.qnum] = self.answerEdit1.text()
        elif self.qnum == 31:
            self.answer[self.qnum] = self.answerEdit1.text()+'&'+self.answerEdit2.text()+'&'+self.answerEdit3.text()+'&'+self.answerEdit4.text()
        elif self.qnum == 32:
            self.answer[self.qnum] = self.answerEdit1.text()
        else:
            self.answer[self.qnum] = self.answer1.currentText()


        # Определяем нажатую кнопку и меняем перменную вопроса
        sender = self.sender()
        if sender.text() == 'Следующий вопрос':
            self.qnum += 1
            self.qNumber.setText(str(self.qnum))

        if sender.text() == 'Предидущий вопрос':
            self.qnum -= 1
            self.qNumber.setText(str(self.qnum))

        if sender.objectName() == 'qNumber':
            self.qnum = int(self.qNumber.text())

        if self.qnum == 1:
            self.previousButton.setEnabled(False)
        elif self.qnum == 32:
            self.nextButton.setEnabled(False)
        else:
            self.previousButton.setEnabled(True)
            self.nextButton.setEnabled(True)

        self.questionLabel.setText("Вопрос №"+str(self.qnum))
        self.questionLabel.adjustSize()

        # Загружаем изображение с вопросом
        self.questionImage.setPixmap(QPixmap('zno/'+str(self.qnum)+'.png'))
        self.questionImage.adjustSize()

        # Определяем номер вопроса и подставляем соответствующие виджеты ответов
        if self.qnum in [21, 22, 23, 24]:
            if self.answer.get(self.qnum) is not None:
                self.answer1.setCurrentText(self.answer[self.qnum][0])
                self.answer2.setCurrentText(self.answer[self.qnum][1])
                self.answer3.setCurrentText(self.answer[self.qnum][2])
                self.answer4.setCurrentText(self.answer[self.qnum][3])
            else:
                self.answer1.setCurrentText('А')
                self.answer2.setCurrentText('А')
                self.answer3.setCurrentText('А')
                self.answer4.setCurrentText('А')
            self.answer1.setVisible(True)
            self.answer2.setVisible(True)
            self.answer3.setVisible(True)
            self.answer4.setVisible(True)
            self.answerEdit1.setVisible(False)
            self.answerEdit2.setVisible(False)
            self.answerEdit3.setVisible(False)
            self.answerEdit4.setVisible(False)
            self.answersLabel.setText('Підберіть відповідний кінець до кожного з чотирьох речень:')
        elif self.qnum in [25, 26]:
            if self.answer.get(self.qnum) is not None:
                self.answerEdit1.setText(self.answer[self.qnum].split('&')[0])
                self.answerEdit2.setText(self.answer[self.qnum].split('&')[1])
            else:
                self.answerEdit1.setText('')
                self.answerEdit2.setText('')
            self.answer1.setVisible(False)
            self.answer2.setVisible(False)
            self.answer3.setVisible(False)
            self.answer4.setVisible(False)
            self.answerEdit1.setVisible(True)
            self.answerEdit2.setVisible(True)
            self.answerEdit3.setVisible(False)
            self.answerEdit4.setVisible(False)
            self.answersLabel.setText('Впишіть відповіді:')
        elif self.qnum in [27, 28, 29, 30]:
            if self.answer.get(self.qnum) is not None:
                self.answerEdit1.setText(self.answer[self.qnum])
            else:
                self.answerEdit1.setText('')
            self.answer1.setVisible(False)
            self.answer2.setVisible(False)
            self.answer3.setVisible(False)
            self.answer4.setVisible(False)
            self.answerEdit1.setVisible(True)
            self.answerEdit2.setVisible(False)
            self.answerEdit3.setVisible(False)
            self.answerEdit4.setVisible(False)
            self.answersLabel.setText('Впишіть відповідь:')
        elif self.qnum == 31:
            if self.answer.get(self.qnum) is not None:
                self.answerEdit1.setText(self.answer[self.qnum].split('&')[0])
                self.answerEdit2.setText(self.answer[self.qnum].split('&')[1])
                self.answerEdit3.setText(self.answer[self.qnum].split('&')[2])
                self.answerEdit4.setText(self.answer[self.qnum].split('&')[3])
            else:
                self.answerEdit1.setText('(;),(;),(;)')
                self.answerEdit2.setText('Test2')
                self.answerEdit3.setText('Test3')
                self.answerEdit4.setText('Test4')
            self.answerEdit1.setVisible(True)
            self.answerEdit2.setVisible(True)
            self.answerEdit3.setVisible(True)
            self.answerEdit4.setVisible(True)
            self.answersLabel.setText('Впишіть відповіді у відповідні поля:')
        elif self.qnum == 32:
            if self.answer.get(self.qnum) is not None:
                self.answerEdit1.setText(self.answer[self.qnum])
            else:
                self.answerEdit1.setText('')
            self.answer1.setVisible(False)
            self.answer2.setVisible(False)
            self.answer3.setVisible(False)
            self.answer4.setVisible(False)
            self.answerEdit1.setVisible(True)
            self.answerEdit2.setVisible(False)
            self.answerEdit3.setVisible(False)
            self.answerEdit4.setVisible(False)
            self.answersLabel.setText('Впишіть відповідь:')
        else:
            if self.answer.get(self.qnum) is not None:
                self.answer1.setCurrentText(self.answer[self.qnum])
                self.statusLabel.setText(str(self.answer))
            else:
                self.answer1.setCurrentText('А')
            self.answer1.setVisible(True)
            self.answer2.setVisible(False)
            self.answer3.setVisible(False)
            self.answer4.setVisible(False)
            self.answerEdit1.setVisible(False)
            self.answerEdit2.setVisible(False)
            self.answerEdit3.setVisible(False)
            self.answerEdit4.setVisible(False)
            self.answersLabel.setText('Оберіть відповідь:')

    def saveResult(self, testNum):
        #Сохраняем данные в файл
        answerFile = open('testresults/'+str(testNum)+str(datetime.datetime.now())+'.txt', 'w')
        answerFile.write(str(self.answer))
        answerFile.close()

    def checkResult(self):
        #Проверка результатов теста
        score = 0
        for key in self.answer.keys():
            if key in [21, 22, 23, 24, 31]:
                for splitedPart in range(4):
                    if self.answer.get(key).split('&')[splitedPart] == self.answersConstDict.get(key).split('&')[splitedPart]:
                        score += 1
            elif key in [25, 26]:
                for splitedPart in range(2):
                    if self.answer.get(key).split('&')[splitedPart] == self.answersConstDict.get(key).split('&')[splitedPart]:
                        score += 1
            elif key in [27, 28, 29, 30]:
                if self.answer.get(key) == self.answersConstDict.get(key):
                    score += 2
            elif key == 32:
                if self.answer.get(key) == self.answersConstDict.get(key):
                    score += 4
            elif key == 33:
                if self.answer.get(key) == self.answersConstDict.get(key):
                    score += 6
            else:
                if self.answer.get(key) == self.answersConstDict.get(key):
                    score += 1
        return score






if __name__ == '__main__':

    app = QApplication(sys.argv)
    eeg_analyzer = App()
    sys.exit(app.exec_())
