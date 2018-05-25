#Загрузка необходимых библиотек
import sys
import os
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QMainWindow
from PyQt5.QtWidgets import QRadioButton, QGroupBox, QFileDialog, QLabel, QSlider, QLineEdit
from PyQt5.QtGui  import QPixmap, QImage, QPainter, QColor
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QPoint
import numpy as np
import cv2
from skimage.measure import label, regionprops


obj_names = ['object 1','object 2','object 3']
obj_palete = [(0, 200, 0), (200, 0, 0), (0, 0, 200)]

#Наследуем от QWidget:
class Example(QMainWindow):
    #Переопределяем конструктор класса:
    def __init__(self, master=None):
        QMainWindow.__init__(self, master)
        self.flag = False
        self.img0 = []
        self.mask_inv = []
        self.initUI()

    #Создание макета проекта:
    def initUI(self):
        self.file_mk_dir = ""
        self.file_dir = ""
        self.filenames = []
        self.file_index = 0
        self.show_markup = True

        self.mainwidget = QWidget(self)
        self.setCentralWidget(self.mainwidget)

        #sizePolicy = QSizePolicy(QSizePolicy.Minimum)
        #self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        #Создание обозначения кнопок:
        self.SelectFileButton = QPushButton("File...")
        self.SelectImDirButton = QPushButton("Image Dir...")
        self.SelectMkDirButton = QPushButton("Markup Dir...")
        self.RecognizeHSVButton = QPushButton("Segment(HSV)")
        #self.RecognizeFCNButton = QPushButton("Segment(FCN)")
        self.FindBlobsButton = QPushButton("Detect")
        self.SaveButton = QPushButton("Save")
        self.LabelCoef = QLabel('мм in 1 pix:')
        self.le_Scale_Coef = QLineEdit()

        self.LabelRadius = QLabel('Brush radius:')
        self.value_Radius = QLineEdit('10')


        self.imageLabel = QLabel()

        self.defaultImWidth = 1500
        self.defaultImHeight = 750

        self.imageLabel.setMinimumWidth(self.defaultImWidth)
        self.imageLabel.setMaximumWidth(self.defaultImWidth)
        self.imageLabel.setMinimumHeight(self.defaultImHeight)
        self.imageLabel.setMaximumHeight(self.defaultImHeight)
        #Связь кнопок с методами их исполнения:
        self.RecognizeHSVButton.clicked.connect(self.startHSVButtonClicked)
        #self.RecognizeFCNButton.clicked.connect(self.startFCNButtonClicked)
        self.FindBlobsButton.clicked.connect(self.FindBlobsButtonClicked)
        self.SelectFileButton.clicked.connect(self.selectFileButtonClicked)
        self.SelectImDirButton.clicked.connect(self.selectImDirButtonClicked)
        self.SelectMkDirButton.clicked.connect(self.selectMkDirButtonClicked)
        self.SaveButton.clicked.connect(self.saveButtonClicked)

        self.hbox = QHBoxLayout()

        #Создание формы кнопок:
        self.hbox.addWidget(self.SelectFileButton)
        self.hbox.addWidget(self.SelectImDirButton)
        self.hbox.addWidget(self.SelectMkDirButton)
        self.hbox.addWidget(self.RecognizeHSVButton)
        #self.hbox.addWidget(self.RecognizeFCNButton)
        self.hbox.addWidget(self.FindBlobsButton)
        self.hbox.addWidget(self.SaveButton)
        self.hbox.addWidget(self.LabelCoef)
        self.hbox.addWidget(self.le_Scale_Coef)

        self.vcontrolboxwidget = QGroupBox()

        self.vcontrolbox = QVBoxLayout()

        self.LeftImButton = QPushButton("< Prev ('a')")
        self.RightImButton = QPushButton("> Next ('d')")
        self.SaveImButton = QPushButton(" Save ('s')")
        self.HideButton = QPushButton(" Hide Mkp ('w')")
        self.SLeftImButton = QPushButton("< Save+Prev ('q')")
        self.SRightImButton = QPushButton("> Save+ Next ('e')")

        self.RightImButton.clicked.connect(self.NextButtonClicked)
        self.LeftImButton.clicked.connect(self.PrevButtonClicked)
        self.SaveImButton.clicked.connect(self.saveButtonClicked)
        self.HideButton.clicked.connect(self.HideButtonClicked)
        self.SLeftImButton.clicked.connect(self.SLeftImButtonClicked)
        self.SRightImButton.clicked.connect(self.SRightImButtonClicked)

        self.vcontrolbox.addWidget(self.SRightImButton)
        self.vcontrolbox.addWidget(self.SLeftImButton)
        self.vcontrolbox.addWidget(self.RightImButton)
        self.vcontrolbox.addWidget(self.LeftImButton)
        self.vcontrolbox.addWidget(self.SaveImButton)
        self.vcontrolbox.addWidget(self.HideButton)

        self.vcontrolbox.addWidget(self.LabelRadius)
        self.vcontrolbox.addWidget(self.value_Radius)

        self.brush_group = QGroupBox('Brush type')
        vbrushbox = QVBoxLayout()
        self.radio_brush_circle = QRadioButton('Circle')
        self.radio_brush_rect = QRadioButton('Rectangle')
        vbrushbox.addWidget(self.radio_brush_circle)
        vbrushbox.addWidget(self.radio_brush_rect)
        self.radio_brush_circle.toggled.connect(self.rbtnbrush)
        self.radio_brush_rect.toggled.connect(self.rbtnbrush)
        self.radio_brush_circle.setChecked(True)
        self.brush_type = 0
        #vbrushbox.addStretch(1)
        self.brush_group.setLayout(vbrushbox)
        self.vcontrolbox.addWidget(self.brush_group)

        self.radio_group = QGroupBox('Objects')
        self.rb_array = []
        index = 0
        vgroupbox = QVBoxLayout()
        for name in obj_names:
            self.rb_array.append(QRadioButton(name))
            vgroupbox.addWidget(self.rb_array[index])
            self.rb_array[index].toggled.connect(self.rbtnstate)
            index += 1
        self.obj_index = 0 #global object index
        self.rb_array[self.obj_index].setChecked(True)
        vgroupbox.addStretch(1)
        self.radio_group.setLayout(vgroupbox)
        self.vcontrolbox.addWidget(self.radio_group)
        self.vcontrolboxwidget.setLayout(self.vcontrolbox)


        #Создание формы приложения:
        self.pixmap = QPixmap()

        self.mainhbox = QHBoxLayout()

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.imageLabel)
        #Создание начального вида окна приложения:
        self.setGeometry(300, 300, 700, 500)
        #self.setMinimumSize(500, 300)
        self.setWindowTitle('Multiclass segmentation tool v.1.0')

        # Лэйблы с описаниями назначения ползунков
        self.LabelHue = QLabel('Hue (H)')
        self.LabelSaturation = QLabel('Saturation (S)')
        self.LabelValue = QLabel('Value (V)')

        self.Coef = 1
        self.le_Scale_Coef.returnPressed.connect(self.valueChangePress)
        self.le_Scale_Coef.setText(str(self.Coef))

        # Создание строк ввода диапазонов и привязка их к методу
        self.value_line_1 = QLineEdit()
        self.value_line_1.returnPressed.connect(self.valueChangePress)
        self.value_line_2 = QLineEdit()
        self.value_line_2.returnPressed.connect(self.valueChangePress)
        self.value_line_3 = QLineEdit()
        self.value_line_3.returnPressed.connect(self.valueChangePress)
        self.value_line_4 = QLineEdit()
        self.value_line_4.returnPressed.connect(self.valueChangePress)
        self.value_line_5 = QLineEdit()
        self.value_line_5.returnPressed.connect(self.valueChangePress)
        self.value_line_6 = QLineEdit()
        self.value_line_6.returnPressed.connect(self.valueChangePress)

        #Создание ползунков:
        self.slider_HueMin = QSlider(Qt.Horizontal, self)
        self.slider_HueMax = QSlider(Qt.Horizontal, self)
        self.slider_SaturationMin = QSlider(Qt.Horizontal, self)
        self.slider_SaturationMax = QSlider(Qt.Horizontal, self)
        self.slider_ValueMin = QSlider(Qt.Horizontal, self)
        self.slider_ValueMax = QSlider(Qt.Horizontal, self)

        self.h1box = QHBoxLayout()
        self.h1box.addWidget(self.LabelHue)
        self.h1box.addWidget(self.value_line_1)
        #self.vbox.addLayout(self.h1box)
        #Ползунку задаются минимальное и максимальное значения шкалы:
        self.slider_HueMin.setMinimum(0)
        self.slider_HueMin.setMaximum(255)
        #Шаг ползунка:
        self.slider_HueMin.setTickInterval(1)
        #Установка начальной позици ползунка:
        self.HueMin = 0
        self.slider_HueMin.setValue(self.HueMin)
        #Связь ползунка с методом, в котором будет  изменятся  значение, которое  нужно изменять:
        self.slider_HueMin.valueChanged.connect(self.valueChange)
        #Установка длины ползунка:
        self.slider_HueMin.setMinimumWidth(200)
        #self.vbox.addWidget(self.slider_HueMin)
        self.h1box.addWidget(self.slider_HueMin)
        self.value_line_1.setText(str(self.HueMin))
        self.value_line_1.setMinimumWidth(25)

        self.h1box.addWidget(self.value_line_2)

        self.slider_HueMax.setMinimum(0)
        self.slider_HueMax.setMaximum(255)
        self.slider_HueMax.setTickInterval(1)
        self.HueMax = 255
        self.slider_HueMax.setValue(self.HueMax)
        self.slider_HueMax.valueChanged.connect(self.valueChange)
        self.slider_HueMax.setMinimumWidth(200)
        self.h1box.addWidget(self.slider_HueMax)
        self.value_line_2.setText(str(self.HueMax))
        self.value_line_2.setMinimumWidth(25)

        self.vbox.addLayout(self.h1box)


        self.h2box = QHBoxLayout()
        self.h2box.addWidget(self.LabelSaturation)
        self.h2box.addWidget(self.value_line_3)

        self.slider_SaturationMin.setMinimum(0)
        self.slider_SaturationMin.setMaximum(255)
        self.slider_SaturationMin.setTickInterval(1)
        self.SaturationMin = 0
        self.slider_SaturationMin.setValue(self.SaturationMin)
        self.slider_SaturationMin.valueChanged.connect(self.valueChange)
        self.slider_SaturationMin.setMinimumWidth(200)
        self.h2box.addWidget(self.slider_SaturationMin)
        self.value_line_3.setText(str(self.SaturationMin))
        self.value_line_3.setMinimumWidth(25)

        self.h2box.addWidget(self.value_line_4)

        self.slider_SaturationMax.setMinimum(0)
        self.slider_SaturationMax.setMaximum(255)
        self.slider_SaturationMax.setTickInterval(1)
        self.SaturationMax = 40
        self.slider_SaturationMax.setValue(self.SaturationMax)
        self.slider_SaturationMax.valueChanged.connect(self.valueChange)
        self.slider_SaturationMax.setMinimumWidth(200)
        self.h2box.addWidget(self.slider_SaturationMax)
        self.value_line_4.setText(str(self.SaturationMax))
        self.value_line_4.setMinimumWidth(25)

        self.vbox.addLayout(self.h2box)

        self.h3box = QHBoxLayout()
        self.h3box.addWidget(self.LabelValue)
        self.h3box.addWidget(self.value_line_5)

        self.slider_ValueMin.setMinimum(0)
        self.slider_ValueMin.setMaximum(255)
        self.slider_ValueMin.setTickInterval(1)
        self.ValueMin = 5
        self.slider_ValueMin.setValue(self.ValueMin)
        self.slider_ValueMin.valueChanged.connect(self.valueChange)
        self.slider_ValueMin.setMinimumWidth(200)
        self.h3box.addWidget(self.slider_ValueMin)
        self.value_line_5.setText(str(self.ValueMin))
        self.value_line_5.setMinimumWidth(25)

        self.h3box.addWidget(self.value_line_6)

        self.slider_ValueMax.setMinimum(0)
        self.slider_ValueMax.setMaximum(255)
        self.slider_ValueMax.setTickInterval(1)
        self.ValueMax = 140
        self.slider_ValueMax.setValue(self.ValueMax)
        self.slider_ValueMax.valueChanged.connect(self.valueChange)
        self.slider_ValueMax.setMinimumWidth(200)
        self.h3box.addWidget(self.slider_ValueMax)
        self.vbox.setSpacing(10)
        self.value_line_6.setText(str(self.ValueMax))
        self.value_line_6.setMinimumWidth(25)

        self.vbox.addLayout(self.h3box)

        self.mainhbox.addLayout(self.vbox)
        self.mainhbox.addWidget(self.vcontrolboxwidget)

        self.mainwidget.setLayout(self.mainhbox)
    def rbtnstate(self):
        radiobutton = self.sender()

        if radiobutton.isChecked():
            index = 0
            for name in obj_names:
                if(radiobutton.text()==name):
                    self.obj_index = index
                index += 1
            self.statusBar().showMessage('Selected index ' + str(self.obj_index))

            #print("Selected country is %s" % (radiobutton.country))

    def rbtnbrush(self):
        radiobutton = self.sender()

        if radiobutton.isChecked():
            index = 0
            if(radiobutton.text() == 'Circle'):
                self.brush_type = 0
            elif (radiobutton.text() == 'Rectangle'):
                self.brush_type = 1
            else:
                self.brush_type = 0
            self.statusBar().showMessage('Brush type ' + str(self.brush_type))

        #Метод открытия изображения:
    def selectFileButtonClicked(self):
        #self.image_file = QFileDialog.getOpenFileName(None, 'Открыть файл', 'D:/data', 'JPG Files(*.jpg);; PNG Files(*.png)')[0]
        self.image_file = QFileDialog.getOpenFileName(None, 'Открыть файл', 'c:/Computer_Vision/MultiClass_Segmentation_Tool/Test_samples',
                                    'JPG Files(*.jpg);; PNG Files(*.png)')[0]
        self.load_image_file(self.image_file)

    #Метод выбора папки с изображениями:
    def selectImDirButtonClicked(self):
        #self.image_file = QFileDialog.getOpenFileName(None, 'Открыть файл', 'D:/data', 'JPG Files(*.jpg);; PNG Files(*.png)')[0]
        self.file_dir = str(QFileDialog.getExistingDirectory(self, "Select directory with images", "D:/Datasets/Students_monitoring"))
        if(self.file_dir !=''):
            self.file_mk_dir = self.file_dir + '/markup'

            self.white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
            self.filenames = []

            for filename in sorted(os.listdir(self.file_dir)):
                is_valid = False
                for extension in self.white_list_formats:
                    if filename.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.filenames.append(filename)

            if(len(self.filenames)==0):
                return

            self.image_file = self.file_dir + '/'+ self.filenames[0]

            self.load_image_file(self.image_file)

            message = "Images directory is successfully selected: " + self.file_dir
            self.statusBar().showMessage(message)

    def selectMkDirButtonClicked(self):
        self.file_mk_dir = str(QFileDialog.getExistingDirectory(self, "Select Markup Directory", "D:/Datasets/Students_monitoring"))
        message = "Markup directory is successfully selected: " + self.file_mk_dir
        self.statusBar().showMessage(message)


    #Метод обработки изображения с помощью HSV:
    def startHSVButtonClicked(self):
        if self.pixmap.isNull() != True:
            self.img0 = cv2.imread(self.image_file, 1)
            self.img = cv2.GaussianBlur(self.img0, (5, 5), 5)
            #Перевод изображения из BGR в цветовую систему HSV:
            self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            #Задание диапозона поиска цветовых пятен:
            self.lower_range = np.array([self.HueMin, self.SaturationMin, self.ValueMin], dtype=np.uint8)
            self.upper_range = np.array([self.HueMax, self.SaturationMax, self.ValueMax], dtype=np.uint8)
            #Маска изображения, выделяющая пятно:
            self.mask = cv2.inRange(self.hsv, self.lower_range, self.upper_range)
            #Инверсия полученной маски:
            #self.mask_inv = cv2.bitwise_not(self.mask)
            self.mask_inv = self.binary_to_color_with_pallete(self.mask, obj_palete[self.obj_index])

            #self.mask_inv = self.mask
            #Наложение инвертированной маски на изображение:
            #self.rez = cv2.bitwise_and(self.img0, self.img0, mask=self.mask_inv)
            self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)
            #cv2.imshow('rez', self.rez)

            """
            # apply threshold
            thresh = 120  # threshold_otsu(image)
            bw = closing(self.mask_inv < thresh, square(3))


            # remove artifacts connected to image border
            cleared = bw.copy()
            clear_border(cleared)
            """

            self.print_image_on_label(self.rez, self.imageLabel)
            self.flag = True
            """
            height, width, channel = self.rez.shape
            bytesPerLine = channel * width
            self.qimg = QImage(self.rez.data, width, height, bytesPerLine, QImage.Format_RGB888)

            #Вывод полученного изображения с маской:
            self.pixmap = QPixmap.fromImage(self.qimg)

            # Вычисляем ширину окна изображения
            w = self.imageLabel.width()
            # Вычисляем высоту окна изображения
            h = self.imageLabel.height()
            self.pixmap = self.pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.imageLabel.setPixmap(self.pixmap)
            """
        else:
            pass

    """
    # Метод обработки изображения с помощью FCN:
    def startFCNButtonClicked(self):
        if self.pixmap.isNull() != True:
            self.img0 = cv2.imread(self.image_file, 1)
            mask_height, mask_width, channel = self.img0.shape
            mask_fcn = prepare_detection_output_from_single_image(self.image_file, 'model_defects_weights.72-0.02-0.22.hdf5', 825, 464)
            mask_fcn = mask_fcn.astype(np.uint8)
            mask_fcn_resized = cv2.resize(mask_fcn, (mask_width, mask_height))
            self.mask_inv = self.binary_to_color_with_pallete(255-mask_fcn_resized)
            #self.rez = cv2.bitwise_and(self.img0, self.img0, mask=self.mask_inv)
            self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)
            self.print_image_on_label(self.rez, self.imageLabel)
    """

    def FindBlobsButtonClicked(self):
        if (self.pixmap.isNull() != True and len(self.mask_inv) != 0):
            # Цвет текста координат:
            self.box_color = (0, 255, 0)

            self.Coef = float(self.le_Scale_Coef.text())
            # label image regions
            #self.mask = cv2.bitwise_not(self.mask_inv)
            self.mask = (self.mask_inv == obj_palete[self.obj_index])
            #self.mask = self.mask.reshape(3, self.mask.shape[0], self.mask.shape[1])
            label_image = label(self.mask[:,:,0])
            # image_label_overlay = label2rgb(label_image, image=self.rez)

            self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)
            #self.rez = cv2.bitwise_and(self.img0, self.img0, mask=self.mask_inv)
            # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            # ax.imshow(image_label_overlay)
            n_all_regions = len(regionprops(label_image))
            ind = 1
            for region in regionprops(label_image):
                # skip small images
                if region.area < 25:
                    continue
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
                cv2.rectangle(self.rez, (minc, minr), (maxc, maxr), self.box_color, 3)
                cv2.putText(self.rez, "%d: %d %d" % (ind, region.area / (self.Coef * self.Coef), region.perimeter / self.Coef),
                            (int(minc+3), int(minr+22)),cv2.FONT_HERSHEY_PLAIN , 1.4, self.box_color, 2)
                # ax.add_patch(rect)
                ind += 1
            # ax.set_axis_off()
            # plt.tight_layout()
            # plt.show()

            # self.rez = cv2.bitwise_and(self.rez, self.rez, mask=rect)
            # cv2.circle(self.rez, ((maxc - minc) / 2, (maxr - minr) / 2), 5, self.color_yellow, 2)
            # cv2.putText(self.rez, "%d" % (region + 1), (((maxc - minc) / 2) + 10, ((maxr - minr) / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_yellow, 2)

            self.print_image_on_label(self.rez, self.imageLabel)
        else:
            pass

    # Метод, в котором изменяется значение отдельных ползунков и связывается с цветом со строкой ввода диапазона:
    def valueChange(self):
        self.HueMin = self.slider_HueMin.value()
        self.value_line_1.setText(str(self.HueMin))
        self.HueMax = self.slider_HueMax.value()
        self.value_line_2.setText(str(self.HueMax))
        self.SaturationMin = self.slider_SaturationMin.value()
        self.value_line_3.setText(str(self.SaturationMin))
        self.SaturationMax = self.slider_SaturationMax.value()
        self.value_line_4.setText(str(self.SaturationMax))
        self.ValueMin = self.slider_ValueMin.value()
        self.value_line_5.setText(str(self.ValueMin))
        self.ValueMax = self.slider_ValueMax.value()
        self.value_line_6.setText(str(self.ValueMax))

    def valueChangePress(self):
        #Изменяет значение диапазона цвета на введённое в строку ввода
        old = self.Coef
        # Считывается новое значение
        new = self.le_Scale_Coef.text()
        # Выполняется проверка на корректность введённого значения
        rez = self.chekForSymb(old, new)
        # Присваивается более коректное значение
        self.Coef = rez
        # Выводится значение в строку ввода
        self.le_Scale_Coef.setText(str(rez))

        # Запоминается прежнее значение на случай ввода недопустимого значения
        old = self.HueMin
        # Считывается новое значение
        new = self.value_line_1.text()
        # Выполняется проверка на корректность введённого значения
        rez = self.chekForSymb(old, new)
        # Присваивается более коректное значение
        self.HueMin = rez
        # Ползунок устанавливается на новое или старое значение
        self.slider_HueMin.setValue(rez)
        # Выводится значение в строку ввода
        self.value_line_1.setText(str(rez))

        old = self.HueMax
        new = self.value_line_2.text()
        rez = self.chekForSymb(old, new)
        self.HueMax = rez
        self.slider_HueMax.setValue(rez)
        self.value_line_2.setText(str(rez))

        old = self.SaturationMin
        new = self.value_line_3.text()
        rez = self.chekForSymb(old, new)
        self.SaturationMin = rez
        self.slider_SaturationMin.setValue(rez)
        self.value_line_3.setText(str(rez))

        old = self.SaturationMax
        new = self.value_line_4.text()
        rez = self.chekForSymb(old, new)
        self.SaturationMax = rez
        self.slider_SaturationMax.setValue(rez)
        self.value_line_4.setText(str(rez))

        old = self.ValueMin
        new = self.value_line_5.text()
        rez = self.chekForSymb(old, new)
        self.ValueMin = rez
        self.slider_ValueMin.setValue(rez)
        self.value_line_5.setText(str(rez))

        old = self.ValueMax
        new = self.value_line_6.text()
        rez = self.chekForSymb(old, new)
        self.ValueMax = rez
        self.slider_ValueMax.setValue(rez)
        self.value_line_6.setText(str(rez))

    def chekForSymb(self, old, new):
        #Возвращает число new в случае его корректности или же old в случае некорректности числа new
        # Запускается генератор множества
        consts = {str(i) for i in range(10)}
        # Перебираются все элементы в new
        for el in new:
            # Если число содержит символы, не являющиеся цифрами, завершается проверка числа на корректность
            if el not in consts:
                # На выход функции подаётся прежнее значение, и функция завершается
                return old
                exit()
        if int(new) < 255:
            return int(new)
        else:
            return 255

    #Метод сохранения изображения:
    def saveButtonClicked(self):
        if (self.flag):
            #File = QFileDialog.getSaveFileName(self, 'Сохранить как', '', 'PNG Files(*.png);; JPG Files(*.jpg)')[0]
            #if File != '':
            res_fname, res_extension = os.path.splitext(self.image_file)
            res_fname = os.path.basename(res_fname)
            res_dirname = os.path.dirname(self.image_file)
            os.makedirs(res_dirname + '/masked', exist_ok=True)
            os.makedirs(res_dirname + '/markup', exist_ok=True)
            target_fname_mask = res_dirname + '/masked/' + res_fname + res_extension
            #target_fname_markup = res_dirname + '/markup/' + res_fname + res_extension
            target_fname_markup = self.file_mk_dir+'/' + res_fname + '.bmp'

            #self.mask = cv2.bitwise_not(self.mask_inv)

            cv2.imwrite(target_fname_markup, self.mask_inv)
            cv2.imwrite(target_fname_mask, self.rez)
            message = "Saved files successfully in folders: " +self.file_mk_dir + ' and ' + res_dirname + '/masked'
            self.statusBar().showMessage(message)

    #Метод присвоения левой кнопке мыши метода рисования:
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.leftclickedflag = True
            self.rightclickedflag = False
            message = "Left click"
        if e.button() == Qt.RightButton:
            self.leftclickedflag = False
            self.rightclickedflag = True
            message = "Right click"
        self.statusBar().showMessage(message)
    def grayscale_to_color(self, grayscale_im):
        im_reshaped = grayscale_im.reshape((grayscale_im.shape[0], grayscale_im.shape[1], 1))
        im_out = np.append(im_reshaped, im_reshaped,axis=2)
        im_out = np.append(im_out, im_reshaped, axis=2)
        return im_out
    def binary_to_color_with_pallete(self, binary_im, pallete_color):
        im_reshaped = binary_im.reshape((binary_im.shape[0], binary_im.shape[1], 1))/255
        im_out = np.append(im_reshaped*(255-pallete_color[0]), im_reshaped*(255-pallete_color[1]),axis=2)
        im_out = np.append(im_out, im_reshaped*(255-pallete_color[2]), axis=2)
        im_out = 255 - im_out
        return im_out

    #Метод рисования дефектов:
    def mouseMoveEvent(self, e):
        if self.leftclickedflag:
            condition, w, h = self.check_paint_field(e, self.flag, self.img0, self.imageLabel)
            if(condition):

                self.mask_inv = self.draw_ellipse_on_mask(e, self.mask_inv, w, h, obj_palete[self.obj_index], int(self.value_Radius.text()), self.brush_type)
                self.rez = (self.img0*(self.mask_inv/255)).astype(np.uint8)
                #self.rez = cv2.bitwise_and(self.img0, self.img0, mask=self.mask_inv)
                self.print_image_on_label(self.rez, self.imageLabel)
                message = "Left move: True"
            else:
                message = "Left move: False"
            self.statusBar().showMessage(message)
            #self.ellips(e)
            #self.imageLabel.setPixmap(QPixmap.fromImage(self.qimg))
        if self.rightclickedflag:
            condition, w, h = self.check_paint_field(e, self.flag, self.img0, self.imageLabel)
            if (condition):
                self.mask_inv = self.draw_ellipse_on_mask(e, self.mask_inv, w, h, (255,255,255),
                                                          int(self.value_Radius.text()), self.brush_type)
                self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)
                self.print_image_on_label(self.rez, self.imageLabel)
                message = "Right move: True"
            else:
                message = "Right move: False"
            self.statusBar().showMessage(message)
            #self.ellips(e)
            #self.imageLabel.setPixmap(QPixmap.fromImage(self.qimg))

    #Метод создания кисти для рисования:
    def ellips(self,e):
        self.paint = QPainter(self.qimg)
        self.paint.setBrush(QColor('black'))
        coords = e.pos()
        geom = self.imageLabel.geometry()
        coords_ed = coords - QPoint(geom.x(), geom.y())
        self.paint.drawEllipse(coords_ed, 10,10)
        self.update()

    # Метод создания кисти для рисования:
    def draw_ellipse_on_mask(self, e, mask_img, label_real_w, label_real_h, draw_color, radius, brush_type):
        coords = e.pos()
        geom = self.imageLabel.geometry()
        coords_ed = coords - QPoint(geom.x(), geom.y())

        mask_height = mask_img.shape[0]
        mask_width = mask_img.shape[1]
        pixmap_height = label_real_h
        pixmap_width = label_real_w

        real_x = int(coords_ed.x()*mask_width/pixmap_width)
        real_y = int(coords_ed.y() * mask_height / pixmap_height)
        if (brush_type == 0):
            cv2.circle(mask_img, (real_x, real_y), radius, draw_color, -1)
        else:
            cv2.rectangle(mask_img, (real_x-radius, real_y-radius), (real_x+radius, real_y+radius), draw_color, -1)
        return mask_img

    def print_image_on_label(self, img, label_widget):
        height, width, channel = img.shape
        bytesPerLine = channel * width
        rgb_cvimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_cvimage.data, width, height, bytesPerLine, QImage.Format_RGB888)

        """
        # Вывод полученного изображения с маской:
        q_pixmap = QPixmap.fromImage(q_img)

        # Вычисляем ширину окна изображения
        w = label_widget.width()
        # Вычисляем высоту окна изображения
        h = label_widget.height()
        q_pixmap = q_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label_widget.setPixmap(q_pixmap)
        """

        self.pixmap = QPixmap(q_img)

        w_p = self.pixmap.width()
        h_p = self.pixmap.height()

        if (w_p > h_p):
            label_new_width = self.defaultImWidth
            label_new_height = h_p * self.defaultImWidth / w_p
        else:
            label_new_width = w_p * self.defaultImHeight / h_p
            label_new_height = self.defaultImHeight

        label_widget.setMinimumWidth(label_new_width)
        label_widget.setMaximumWidth(label_new_width)
        label_widget.setMinimumHeight(label_new_height)
        label_widget.setMaximumHeight(label_new_height)

        # Вычисляем ширину окна изображения
        w = label_widget.width()
        # Вычисляем высоту окна изображения
        h = label_widget.height()
        self.pixmap = self.pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label_widget.setPixmap(self.pixmap)

        return True

    def check_paint_field(self, e, rec_flag, img, label_widget):
        if(rec_flag):
            img_height, img_width, img_channel = img.shape
            geom = label_widget.geometry()

            scale_x_w = geom.width()
            scale_y_h = geom.height()

            scale_ratio_img = img_height/img_width
            scale_ratio_label = geom.height() / geom.width()
            if(scale_ratio_img > scale_ratio_label):
                scale_x_w = geom.height()/scale_ratio_img
            if (scale_ratio_img < scale_ratio_label):
                scale_y_h = geom.width() * scale_ratio_img

            coords = e.pos()

            x_max = geom.x() + scale_x_w
            y_max = geom.y() + scale_y_h
            condition_label = coords.x() > geom.x() and coords.x() < x_max and coords.y() > geom.y() and coords.y() < y_max
            if(condition_label):
                return True, scale_x_w, scale_y_h
            #coords_ed = coords - QPoint(geom.x(), geom.y())

        return False, 0, 0
    def NextButtonClicked(self):
        if(len(self.filenames)>self.file_index+1):
            self.file_index += 1
            self.image_file = self.file_dir + '/'+self.filenames[self.file_index]
            self.load_image_file(self.image_file)
            self.statusBar().showMessage('> next image')
        else:
            self.statusBar().showMessage('no next image')

    def PrevButtonClicked(self):
        if (self.file_index > 0):
            self.file_index -= 1
            self.image_file = self.file_dir + '/' + self.filenames[self.file_index]
            self.load_image_file(self.image_file)
            self.statusBar().showMessage('< previous image')
        else:
            self.statusBar().showMessage('no previous image')

    #Загрузка изображения на форму
    def load_image_file(self, file_name):

        self.img0 = cv2.imread(file_name, 1)
        res_fname, res_extension = os.path.splitext(self.image_file)
        res_fname = os.path.basename(res_fname)
        mask_path = self.file_mk_dir+'/' + res_fname+'.bmp'
        self.mask_inv = cv2.imread(mask_path,1)
        if(type(self.mask_inv) is np.ndarray):
            self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)
        else:
            self.rez = self.img0
            self.mask_inv = np.ones(self.img0.shape, dtype='uint8')*255

        self.flag = True
        # self.rez = cv2.bitwise_and(self.img0, self.img0, mask=self.mask_inv)
        self.print_image_on_label(self.rez, self.imageLabel)

    # Загрузка изображения на форму
    def load_just_image_file(self, file_name):
        self.img0 = cv2.imread(file_name, 1)
        self.rez = self.img0
        self.print_image_on_label(self.rez, self.imageLabel)

    def HideButtonClicked(self):
        if self.show_markup:
            self.show_markup = False
            self.flag = False
            self.load_just_image_file(self.image_file)
        else:
            self.show_markup = True
            self.flag = True
            self.load_image_file(self.image_file)

    def SLeftImButtonClicked(self):
        self.saveButtonClicked()
        self.PrevButtonClicked()

    def SRightImButtonClicked(self):
        self.saveButtonClicked()
        self.NextButtonClicked()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_A:
            self.PrevButtonClicked()
        elif e.key() == Qt.Key_D:
            self.NextButtonClicked()
        elif e.key() == Qt.Key_S:
            self.saveButtonClicked()
        elif e.key() == Qt.Key_W:
            self.HideButtonClicked()
        elif e.key() == Qt.Key_Q:
            self.SLeftImButtonClicked()
        elif e.key() == Qt.Key_E:
            self.SRightImButtonClicked()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    ex.move(0, 0)
    ex.show()

    sys.exit(app.exec_())