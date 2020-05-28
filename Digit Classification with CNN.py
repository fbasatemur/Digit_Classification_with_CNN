# -*- coding: utf-8 -*-
"""
Created on Thu May 21 02:23:02 2020

@author: fbasatemur
"""

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QImage,QPainter, QPen
from PyQt5.QtCore import Qt, QPoint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# opencv use for image resize 
import cv2                          

# CNN
import keras 
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

#%% preprocess

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# plt.figure()
# plt.imshow(x_train[10], cmap = "gray")
# plt.axis("off")
# plt.grid(False)

img_rows = 28
img_cols = 28

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)


#%% normalization

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

#x_train_norm = np.linalg.norm(x_train)
#x_train = x_train / x_train_norm

x_train = x_train / 255
x_test = x_test / 255

number_of_classes = 10
y_train = keras.utils.to_categorical(y_train, number_of_classes)
y_test = keras.utils.to_categorical(y_test, number_of_classes)


#%% CNN

model_list = []
score_list = []

batch_size = 256
epochs = 5

# 16,32,64 -> model 1
# 8,16,32  -> model 2
filter_numbers = np.array([[16,32,64],[8,16,32]])

for i in range(2):
      print(filter_numbers[i])
      model = Sequential()
      model.add(Conv2D(filter_numbers[i,0], kernel_size = (3,3), activation = "relu", input_shape = input_shape))
      model.add(Conv2D(filter_numbers[i,1], kernel_size = (3,3), activation = "relu"))
      model.add(MaxPooling2D(pool_size = (2,2)))
      model.add(Dropout(0.25))
    
      model.add(Flatten())
      model.add(Dense(filter_numbers[i,2], activation = "relu"))
      model.add(Dropout(0.5))
      model.add(Dense(number_of_classes, activation = "softmax"))

      model.compile(loss = keras.losses.categorical_crossentropy, 
                    optimizer = keras.optimizers.Adadelta(), metrics = ["accuracy"])
      
      history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))
 
      score = model.evaluate(x_test, y_test, verbose = 0)
      print("Model: {} Test Loss: {}".format(i+1, score[0]))
      print("Model: {} Test Accuracy: {}".format(i+1, score[1]))
      model_list.append(model)
      score_list.append(score)
      
      model.save("model" + str(i+1) + ".h5")


#%% model load

model1 = load_model("model1.h5")
model2 = load_model("model2.h5")

#%% Canvas
class Canvas(QMainWindow):
      
      def __init__(self):
            super().__init__()
      
            self.width = 400
            self.height = 400
            self.setWindowTitle("Draw Digit")
            self.setGeometry(100, 100, self.width, self.height)
            self.setWindowIcon(QIcon("write_pencil-512.png")) 
            
            
            self.image = QImage(self.size(), QImage.Format_RGB32)
            self.image.fill(Qt.black)
            
            self.lastPoint = QPoint()
            self.drawing = False
            
            # image array
            self.image_np = np.zeros([self.width, self.height])
            
            button1 = QPushButton("Ok", self)
            button1.move(2,2)
            button1.clicked.connect(self.enterFunc)
            
            
      def paintEvent(self, event):
            canvasPainter = QPainter(self)
            canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
      
      def enterFunc(self):
            ptr = self.image.constBits()
            ptr.setsize(self.image.byteCount())
            
            self.image_np = np.array(ptr).reshape(self.width, self.height, 4)
            self.image_np = self.image_np[:,:,0]
            self.image_np = self.image_np / 255.0
            
            if np.sum(self.image_np) == 0:
                  print("please write a digit")
            else:
                  plt.figure(figsize = (1,1), dpi = 200)
                  plt.imshow(self.image_np, cmap="gray")
                  plt.axis("off")
                  plt.grid(False)
                  plt.savefig("./input_Image.png")
                  
                  self.close()
                  
      def mousePressEvent(self, event):
            
            if event.button() == Qt.LeftButton:
                  self.lastPoint = event.pos()
                  self.drawing = True
                  
                  print(self.lastPoint)
      
      def mouseMoveEvent(self, event):
            
            if (event.buttons() == Qt.LeftButton) & self.drawing:
                  painter = QPainter(self.image)
                  painter.setPen(QPen(Qt.white, 15, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                  painter.drawLine(self.lastPoint, event.pos())
                  self.lastPoint = event.pos()
                  self.update()
      
      def mouseReleaseEvent(self, event):
            
            if event.button == Qt.LeftButton:
                  self.drawing = False
                  
      
#%% GUI

class Window(QMainWindow):
      
      def __init__(self):
            super().__init__()
            
            # main window
            self.width = 1280
            self.height = 640
            
            self.setWindowTitle("Digit Classification")
            self.setGeometry(200, 200, self.width, self.height)
            # self.setWindowIcon(QIcon("icon.png"))            
            
            self.createCanvas = Canvas()
            
            self.tabWidget()
            self.widgets()
            self.layouts()
            
            self.show()
            
      def tabWidget(self):
            self.tabs = QTabWidget()
            self.setCentralWidget(self.tabs)
            
            self.tab1 = QWidget()
            self.tab2 = QWidget()
            
            self.tabs.addTab(self.tab1, "Classification")
            self.tabs.addTab(self.tab2, "Parameters")
            
      
      def widgets(self):
            
            # tab1 left
            self.drawCanvas = QPushButton("Draw Canvas")
            self.drawCanvas.clicked.connect(self.drawCanvasFunc)
            
            self.openCanvas = QPushButton("Open Canvas")
            self.openCanvas.clicked.connect(self.openCanvasFunc)
            
            self.inputImage = QLabel(self)
            # self.inputImage.setPixmap(QPixmap("input_Image.png.png"))
            
            self.searchText = QLabel("Real Number: ")            
            
            self.searchEntry = QLineEdit()
            self.searchEntry.setPlaceholderText("Which number do you write ?")
            
            # tab1 left Middle
            self.methodSelection = QComboBox(self)
            self.methodSelection.addItems(["model1", "model2"])
            
            self.noiseText = QLabel("Add Noise: %" + "0")
            self.noiseSlider = QSlider(Qt.Horizontal)
            self.noiseSlider.setMinimum(0)
            self.noiseSlider.setMaximum(100)
            self.noiseSlider.setTickPosition(QSlider.TicksBelow)
            self.noiseSlider.valueChanged.connect(self.noiseSliderFunc)      
            
            self.remember = QCheckBox("Save Result", self)
            
            self.predict = QPushButton("Predict")
            self.predict.clicked.connect(self.predictionFunc)
            
            #tab1 right middle 
            self.outputImage = QLabel(self)
            self.outputLabel = QLabel("", self)
            self.outputLabel.setAlignment(Qt.AlignCenter)
            
            #tab1 right
            self.resultTable = QTableWidget()
            self.resultTable.setColumnCount(2)
            self.resultTable.setRowCount(10)
            self.resultTable.setHorizontalHeaderItem(0, QTableWidgetItem("Label(Class)"))
            self.resultTable.setHorizontalHeaderItem(1, QTableWidgetItem("Probability"))
            
            #tab2 method1
            self.parameter_list1 = QListWidget(self)
            self.parameter_list1.addItems(["batch size = 256", "epochs = 5", "image rows = 28",
                                           "image cols = 28", "Filter = [16,32,64]", 
                                           "Activation Function  = ReLu", 
                                           "loss = categorical cross entropy",
                                           "optimizer = AdaDelta", "metrics = accuracy"])
            
            #tab2 method2
            self.parameter_list2 = QListWidget(self)
            self.parameter_list2.addItems(["batch size = 256", "epochs = 5", "image rows = 28",
                                           "image cols = 28", "Filter = [8,16,32]", 
                                           "Activation Function  = ReLu", 
                                           "loss = categorical cross entropy",
                                           "optimizer = AdaDelta", "metrics = accuracy"])
            
            
            
            
      def drawCanvasFunc(self):
            self.createCanvas.show()
      
      def openCanvasFunc(self):
            self.inputImage.setPixmap(QPixmap("input_Image.png"))

      def noiseSliderFunc(self):
            val = self.noiseSlider.value()
            self.noiseText.setText("Add noise: % "+ str(val))

      def predictionFunc(self):
            save_str = ""
            
            real_entry = self.searchEntry.text()
            save_str = save_str + " real entry: " + str(real_entry) + ", "
            
            # CNN model selection
            model_name = self.methodSelection.currentText()
            
            if model_name == "model1":
                  model = load_model("model1.h5")
            elif model_name == "model2":
                  model = load_model("model2.h5")
            else:
                  print("Error")
            
            save_str += "model name: " + str(model_name) + ", "            
            
            # slider noise 
            noise_val = self.noiseSlider.value()
            
            if noise_val != 0:
                  noise_array = np.random.randint(0, noise_val, (28,28))/100
            
            else:
                  noise_array = np.zeros([28,28])
                  
            save_str += "noise value: " + str(noise_val) + ", "
            print(save_str)
            
            # load image as numpy 
            image_array = mpimg.imread("input_Image.png")[26:175,26:175,0]
            # plt.imshow(image_array, cmap = "gray")
            resized_image_array = cv2.resize(image_array, dsize = (28,28), interpolation = cv2.INTER_CUBIC)
            
            # add noise
            resized_image_array += noise_array
            
            plt.imshow(resized_image_array, cmap = "gray")
            plt.title("added noise and resize")
            
            # predict
            result = model.predict(resized_image_array.reshape(1,28,28,1))
            QMessageBox.information(self, "information", "Classification is completed.")
            predicted_class = np.argmax(result)
            print("Prediction:", predicted_class)
            
            save_str += "Prediction class: " + str(predicted_class)
            
            # save result
            if self.remember.isChecked():
                  text_file = open("Output.txt", "w")
                  text_file.write(save_str)
                  text_file.close()
            else:
                  print("not save")
            
            self.outputImage.setPixmap(QPixmap("output_images\\" + str(predicted_class) + ".png"))
            self.outputLabel.setText("Real: " + str(real_entry) + " and Predicted:" + str(predicted_class))
            
            
            # set result
            for row in range(10):
                  self.resultTable.setItem(row, 0, QTableWidgetItem(str(row)))
                  self.resultTable.setItem(row, 1, QTableWidgetItem(str(np.round(result[0][row], 5))))
            
      def layouts(self):
            
            #tab1 layout
            self.mainLayout = QHBoxLayout()
            self.leftLayout = QFormLayout()
            self.leftMiddleLayout = QFormLayout()
            self.rightMiddleLayout = QFormLayout()
            self.rightLayout = QFormLayout()
            
            # left
            self.leftLayoutGroupBox = QGroupBox("Input Image")
            self.leftLayout.addRow(self.drawCanvas)
            self.leftLayout.addRow(self.openCanvas)
            self.leftLayout.addRow(self.inputImage)
            self.leftLayout.addRow(self.searchText)
            self.leftLayout.addRow(self.searchEntry)
            self.leftLayoutGroupBox.setLayout(self.leftLayout)
            
            # left middle
            self.leftMiddleLayoutGroupBox = QGroupBox("Settings")
            self.leftMiddleLayout.addRow(self.methodSelection)
            self.leftMiddleLayout.addRow(self.noiseText)
            self.leftMiddleLayout.addRow(self.noiseSlider)
            self.leftMiddleLayout.addRow(self.remember)
            self.leftMiddleLayout.addRow(self.predict)
            self.leftMiddleLayoutGroupBox.setLayout(self.leftMiddleLayout)
            
            # right middle
            self.rightMiddleLayoutGroupBox = QGroupBox("Output")
            self.rightMiddleLayout.addRow(self.outputImage)
            self.rightMiddleLayout.addRow(self.outputLabel)  
            self.rightMiddleLayoutGroupBox.setLayout(self.rightMiddleLayout)
            
            # right 
            self.rightLayoutGroupBox = QGroupBox("Result")
            self.rightLayout.addRow(self.resultTable)
            self.rightLayoutGroupBox.setLayout(self.rightLayout)
            
            # tab1 main Layout
            self.mainLayout.addWidget(self.leftLayoutGroupBox, 25)
            self.mainLayout.addWidget(self.leftMiddleLayoutGroupBox, 25)
            self.mainLayout.addWidget(self.rightMiddleLayoutGroupBox, 25)
            self.mainLayout.addWidget(self.rightLayoutGroupBox, 25)
            self.tab1.setLayout(self.mainLayout)
            
            #tab2 layout
            self.tab2Layout = QHBoxLayout()
            self.tab2Method1Layout = QFormLayout()
            self.tab2Method2Layout = QFormLayout()
            
            # tab2 Method1 Layout
            self.tab2Method1LayoutGroupBox = QGroupBox("Method1")
            self.tab2Method1Layout.addRow(self.parameter_list1)
            self.tab2Method1LayoutGroupBox.setLayout(self.tab2Method1Layout)
            
            # tab2 Method2 Layout
            self.tab2Method2LayoutGroupBox = QGroupBox("Method2")
            self.tab2Method2Layout.addRow(self.parameter_list2)
            self.tab2Method2LayoutGroupBox.setLayout(self.tab2Method2Layout)
            
            # tab2 main Layout
            self.tab2Layout.addWidget(self.tab2Method1LayoutGroupBox, 50)
            self.tab2Layout.addWidget(self.tab2Method2LayoutGroupBox, 50)
            self.tab2.setLayout(self.tab2Layout)
            
            
            
window = Window();





