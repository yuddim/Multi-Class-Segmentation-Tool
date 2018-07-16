#Загрузка необходимых библиотек
import sys
import os
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QMainWindow
from PyQt5.QtWidgets import QRadioButton, QGroupBox, QFileDialog, QLabel, QSlider, QLineEdit
from PyQt5.QtGui  import QPixmap, QImage, QPainter, QColor, QCursor
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QPoint, QObject
import numpy as np
import cv2
from skimage.measure import label, regionprops

obj_names = ['object 1','object 2','object 3']
obj_palete = [(0, 200, 0), (200, 0, 0), (0, 0, 200)]

#Наследуем от QMainWindow:
class SegmenationTool(QMainWindow):
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
        self.condition = False
        self.leftclickedflag = False
        self.rightclickedflag = False

        self.mainwidget = QWidget(self)
        self.setCentralWidget(self.mainwidget)

        #Создание обозначения кнопок:
        self.SelectFileButton = QPushButton("File...")
        self.SelectImDirButton = QPushButton("Image Dir...")
        self.SelectMkDirButton = QPushButton("Markup Dir...")
        self.RecognizeHSVButton = QPushButton("Segment(HSV)")
        #self.RecognizeFCNButton = QPushButton("Segment(FCN)")
        self.FindBlobsButton = QPushButton("Detect")
        #self.CheckPathButton = QPushButton("Check Path")#CheckNumObjectsPath
        self.SaveButton = QPushButton("Save")
        self.GotoButton = QPushButton("Go to image id:")
        self.le_img_id = QLineEdit()

        self.LabelCoef = QLabel('мм in 1 pix:')
        self.le_Scale_Coef = QLineEdit()

        self.LabelRadius = QLabel("Brush radius ('+','-'):")
        self.value_Radius = QLineEdit('10')
        self.radius = int(self.value_Radius.text())

        self.imageLabel = QLabel()

        self.defaultImWidth = 1500
        self.defaultImHeight = 750

        self.imageLabel.setMinimumWidth(self.defaultImWidth)
        self.imageLabel.setMaximumWidth(self.defaultImWidth)
        self.imageLabel.setMinimumHeight(self.defaultImHeight)
        self.imageLabel.setMaximumHeight(self.defaultImHeight)
        #Связь кнопок с методами их исполнения:
        self.RecognizeHSVButton.clicked.connect(self.startHSVButtonClicked)

        self.FindBlobsButton.clicked.connect(self.FindBlobsEasyButtonClicked)
        self.SelectFileButton.clicked.connect(self.selectFileButtonClicked)
        self.SelectImDirButton.clicked.connect(self.selectImDirButtonClicked)
        self.SelectMkDirButton.clicked.connect(self.selectMkDirButtonClicked)
        self.SaveButton.clicked.connect(self.saveButtonClicked)
        self.GotoButton.clicked.connect(self.GotoButtonClicked)

        #self.CheckPathButton.clicked.connect(self.CheckNumObjectsPath)

        self.hbox = QHBoxLayout()

        #Создание формы кнопок:
        self.hbox.addWidget(self.SelectFileButton)
        self.hbox.addWidget(self.SelectImDirButton)
        self.hbox.addWidget(self.SelectMkDirButton)
        self.hbox.addWidget(self.RecognizeHSVButton)
        #self.hbox.addWidget(self.RecognizeFCNButton)
        self.hbox.addWidget(self.FindBlobsButton)
        self.hbox.addWidget(self.SaveButton)
        self.hbox.addWidget(self.GotoButton)
        self.hbox.addWidget(self.le_img_id)

        #self.hbox.addWidget(self.CheckPathButton)

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
        self.vbrushbox = QVBoxLayout()
        #vbrushbox.setMouseTracking(True)
        self.radio_brush_circle = QRadioButton('Circle')
        self.radio_brush_rect = QRadioButton('Rectangle')

        self.vbrushbox.addWidget(self.radio_brush_circle)
        self.vbrushbox.addWidget(self.radio_brush_rect)
        self.radio_brush_circle.toggled.connect(self.rbtnbrush)
        self.radio_brush_rect.toggled.connect(self.rbtnbrush)
        self.radio_brush_circle.setChecked(True)
        self.brush_type = 0
        #vbrushbox.addStretch(1)
        self.brush_group.setLayout(self.vbrushbox)
        self.vcontrolbox.addWidget(self.brush_group)

        self.radio_group = QGroupBox('Objects')
        self.rb_array = []
        index = 0
        self.vgroupbox = QVBoxLayout()
        for name in obj_names:
            self.rb_array.append(QRadioButton(name))
            self.vgroupbox.addWidget(self.rb_array[index])
            self.rb_array[index].toggled.connect(self.rbtnstate)
            index += 1
        self.obj_index = 0 #global object index
        self.rb_array[self.obj_index].setChecked(True)
        self.vgroupbox.addStretch(1)
        self.radio_group.setLayout(self.vgroupbox)
        self.radio_group.setMouseTracking(True)
        #self.vcontrolbox.addLayout(self.vgroupbox)
        self.vcontrolbox.addWidget(self.radio_group)
        #self.vcontrolboxwidget.setLayout(self.vcontrolbox)


        #Создание формы приложения:
        self.pixmap = QPixmap()

        self.mainhbox = QHBoxLayout()

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.imageLabel)
        #Создание начального вида окна приложения:
        self.setGeometry(300, 300, 700, 500)
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
        self.mainhbox.addLayout(self.vcontrolbox)

        self.mainwidget.setLayout(self.mainhbox)
        self.setMouseTracking(True)

    def setMouseTracking(self, flag):
        def recursive_set(parent):
            for child in parent.findChildren(QObject):
                try:
                    child.setMouseTracking(flag)
                    #print(str(child))
                except:
                    pass
                recursive_set(child)

        QWidget.setMouseTracking(self, flag)
        recursive_set(self)

    def check_and_repaint_cursor(self, e):
        condition, w, h, sc = self.check_paint_field(e, self.flag, self.img0, self.imageLabel)
        self.repaint_cursor(sc)


    def repaint_cursor(self, scale_coef):
        if(not self.condition):
            self.draw_cursor_default()
        else:
            radius = int(int(self.value_Radius.text())*scale_coef)
            r = obj_palete[self.obj_index][0]
            g = obj_palete[self.obj_index][1]
            b = obj_palete[self.obj_index][2]
            if(self.radio_brush_circle.isChecked()):
                self.draw_cursor_circle(radius, (b, g, r, 128))
            else:
                self.draw_cursor_rectangle(radius, (b, g, r, 128))
        return

    def draw_cursor_circle(self, radius, color):
        diameter = 2*radius
        self.m_LPixmap = QPixmap(diameter,diameter)
        self.m_LPixmap.fill(Qt.transparent)
        self.painter = QPainter(self.m_LPixmap)
        self.brush_color = QColor(color[0], color[1], color[2], color[3])
        self.painter.setPen(Qt.NoPen)
        self.painter.setBrush(self.brush_color)
        self.painter.drawEllipse(0,0,diameter,diameter)
        self.m_cursor = QCursor(self.m_LPixmap)
        self.setCursor(self.m_cursor)
        return

    def draw_cursor_rectangle(self, radius, color):
        width = 2 * radius
        self.m_LPixmap = QPixmap(width, width)
        self.m_LPixmap.fill(Qt.transparent)
        self.painter = QPainter(self.m_LPixmap)
        self.brush_color = QColor(color[0], color[1], color[2], color[3])
        self.painter.setPen(Qt.NoPen)
        self.painter.setBrush(self.brush_color)
        self.painter.drawRect(0, 0, width, width)
        self.m_cursor = QCursor(self.m_LPixmap)
        self.setCursor(self.m_cursor)
        return

    def draw_cursor_default(self):
        m_cursor = QCursor()
        self.setCursor(m_cursor)
        return

    def rbtnstate(self):
        radiobutton = self.sender()

        if radiobutton.isChecked():
            index = 0
            for name in obj_names:
                if(radiobutton.text()==name):
                    self.obj_index = index
                index += 1
            self.statusBar().showMessage('Selected index ' + str(self.obj_index))


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
        self.image_file = QFileDialog.getOpenFileName(None, 'Открыть файл', 'c:/Computer_Vision/MultiClass_Segmentation_Tool/Test_samples',
                                    'JPG Files(*.jpg);; PNG Files(*.png)')[0]
        if(self.image_file != ''):
            self.load_image_file(self.image_file)

    #Метод выбора папки с изображениями:
    def selectImDirButtonClicked(self):
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
            self.setWindowTitle('Multiclass segmentation tool v.1.0'+' | File: ' + self.filenames[0])
            self.le_img_id.setText(str(0))

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

            #Наложение инвертированной маски на изображение:
            self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)

            self.print_image_on_label(self.rez, self.imageLabel)
            self.flag = True
        else:
            pass

    def CheckNumObjectsPath(self):
        self.file_index = 0
        ind = 0
        output_file = open('statistics\\folder_statistics_ethalon.txt', 'w')
        out_text = "n_obj\tfilename\n"
        output_file.write(out_text)

        for ind in range(len(self.filenames)):
            n_objects = self.FindBlobsEasyButtonClicked()
            out_text = "%d\t%s\n" % (n_objects, self.filenames[self.file_index])
            output_file.write(out_text)
            self.NextButtonClicked()
        output_file.close()

    def FindBlobsEasyButtonClicked(self):
        if (self.pixmap.isNull() != True and len(self.mask_inv) != 0):

            # Цвет текста координат:
            self.box_color = (0, 255, 0)

            self.Coef = float(self.le_Scale_Coef.text())
            # label image regions
            self.mask = (self.mask_inv == obj_palete[self.obj_index])
            label_image = label(self.mask[:, :, 0])

            self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)
            n_all_regions = len(regionprops(label_image))
            ind = 1
            im_width = self.rez.shape[0]
            for region in regionprops(label_image):
                # skip small images
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox

                S = (maxc - minc) * (maxr - minr)

                if (S < 200):
                    continue

                cv2.rectangle(self.rez, (minc, minr), (maxc, maxr), self.box_color, 1)
                cv2.putText(self.rez, "%d: %d %d %d" % (ind, S,  region.area / (self.Coef * self.Coef), region.perimeter / self.Coef),
                            (int(minc+3), int(minr+22)),cv2.FONT_HERSHEY_PLAIN , 1.4, self.box_color, 1)

                ind += 1

            self.print_image_on_label(self.rez, self.imageLabel)
            return ind
        else:
            return 0
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

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.leftclickedflag = False
            message = "Left click release"
        if e.button() == Qt.RightButton:
            self.rightclickedflag = False
            message = "Right click release"
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
        coords = e.pos()
        old_condition = self.condition
        self.condition = self.is_in_field(e, self.imageLabel)
        self.statusBar().showMessage(str(coords.x()) + ' ' + str(coords.y())+' '+str(self.condition))
        if (old_condition != self.condition and self.flag):
            sc = self.calc_scale_coef(self.img0, self.imageLabel)
            self.repaint_cursor(sc)

        if self.leftclickedflag:
            condition, w, h, sc = self.check_paint_field(e, self.flag, self.img0, self.imageLabel)
            if(condition):
                self.mask_inv = self.draw_ellipse_on_mask(e, self.mask_inv, w, h, obj_palete[self.obj_index], int(self.value_Radius.text()), self.brush_type)
                self.rez = (self.img0*(self.mask_inv/255)).astype(np.uint8)
                #self.rez = cv2.bitwise_and(self.img0, self.img0, mask=self.mask_inv)
                self.print_image_on_label(self.rez, self.imageLabel)
                message = "Left move: True"
            else:
                message = "Left move: False"

            self.statusBar().showMessage(message)

        if self.rightclickedflag:
            condition, w, h, sc = self.check_paint_field(e, self.flag, self.img0, self.imageLabel)
            if (condition):
                self.mask_inv = self.draw_ellipse_on_mask(e, self.mask_inv, w, h, (255,255,255),
                                                          int(self.value_Radius.text()), self.brush_type)
                self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)
                self.print_image_on_label(self.rez, self.imageLabel)
                message = "Right move: True"
            else:
                message = "Right move: False"
            self.statusBar().showMessage(message)

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
                scale_coef = geom.height() / img_height
            if (scale_ratio_img <= scale_ratio_label):
                scale_y_h = geom.width() * scale_ratio_img
                scale_coef = geom.width() / img_width
            coords = e.pos()

            x_max = geom.x() + scale_x_w
            y_max = geom.y() + scale_y_h
            condition_label = coords.x() > geom.x() and coords.x() < x_max and coords.y() > geom.y() and coords.y() < y_max
            if(condition_label):
                return True, scale_x_w, scale_y_h, scale_coef

        return False, 0, 0, 0

    def is_in_field(self, e, label_widget):
        geom = label_widget.geometry()

        coords = e.pos()

        x_max = geom.x() + geom.width()
        y_max = geom.y() + geom.height()
        condition_label = coords.x() > geom.x() and coords.x() < x_max and coords.y() > geom.y() and coords.y() < y_max
        if(condition_label):
            return True
        return False

    def calc_scale_coef(self, img, label_widget):
        img_height, img_width, img_channel = img.shape
        scale_coef = 1
        geom = label_widget.geometry()

        scale_x_w = geom.width()
        scale_y_h = geom.height()
        if(img_width!=0 and img_height!=0):
            scale_ratio_img = img_height/img_width
            scale_ratio_label = geom.height() / geom.width()
            if(scale_ratio_img > scale_ratio_label):
                scale_coef = geom.height() / img_height
            if (scale_ratio_img <= scale_ratio_label):
                scale_coef = geom.width() / img_width

        return scale_coef


    def NextButtonClicked(self):
        if(len(self.filenames)>self.file_index+1):
            self.file_index += 1
            self.image_file = self.file_dir + '/'+self.filenames[self.file_index]
            self.setWindowTitle('Multiclass segmentation tool v.1.0' + ' | File: ' + self.filenames[self.file_index])
            self.le_img_id.setText(str(self.file_index))
            self.load_image_file(self.image_file)
            self.statusBar().showMessage('> next image')
        else:
            self.statusBar().showMessage('no next image')

    def PrevButtonClicked(self):
        if (self.file_index > 0):
            self.file_index -= 1
            self.image_file = self.file_dir + '/' + self.filenames[self.file_index]
            self.setWindowTitle('Multiclass segmentation tool v.1.0' + ' |  File: ' + self.filenames[self.file_index])
            self.le_img_id.setText(str(self.file_index))
            self.load_image_file(self.image_file)
            self.statusBar().showMessage('< previous image')
        else:
            self.statusBar().showMessage('no previous image')

    def GotoButtonClicked(self):
        self.file_index = int(self.le_img_id.text())
        if (self.file_index >= 0 and len(self.filenames)>self.file_index):
            self.image_file = self.file_dir + '/' + self.filenames[self.file_index]
            self.setWindowTitle('Multiclass segmentation tool v.1.0' + ' |  File: ' + self.filenames[self.file_index])
            self.le_img_id.setText(str(self.file_index))
            self.load_image_file(self.image_file)
            self.statusBar().showMessage('find image')
        else:
            self.statusBar().showMessage('no image')


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

    def IncreaseRadius(self, step):
        if(self.radius < 1000 and self.flag):
            self.radius += step
            self.value_Radius.setText(str(self.radius))
            sc = self.calc_scale_coef(self.img0, self.imageLabel)
            self.repaint_cursor(sc)

    def DerceaseRadius(self, step):
        if(self.radius > step and self.flag):
            self.radius -= step
            self.value_Radius.setText(str(self.radius))
            sc = self.calc_scale_coef(self.img0, self.imageLabel)
            self.repaint_cursor(sc)

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
        elif e.key() == Qt.Key_Plus:
            self.IncreaseRadius(1)
        elif e.key() == Qt.Key_Minus:
            self.DerceaseRadius(1)

    def wheelEvent(self, event):
        numDegrees = event.angleDelta().y() / 8
        numSteps = numDegrees / 15.0
        if(numSteps > 0):
            self.IncreaseRadius(2)
        elif (numSteps < 0):
            self.DerceaseRadius(2)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SegmenationTool()
    ex.move(0, 0)
    ex.show()

    sys.exit(app.exec_())