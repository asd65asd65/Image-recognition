# USAGE
# python train_network.py --dataset images --model santa_cat_dog.model

# import matplotlib
# matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from LeNet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

import fileList

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-m", "--model", required=True,help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

EPOCHS = 80
INIT_LR = 1e-3
BS = 100
# EPOCHS代表要訓練的次數
# INIT_LR代表初始化學習率(Initial Learn Rate)
# BS(Batch Size)代表每次訓練的資料量
ClassNum = 4
#ClassNum類別數量
RS = 30
#RS(Random Seed)隨機種子

print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(RS)
# random.seed(42)是為了取得可重現的訓練結果，而特意設定的隨機種子
random.shuffle(imagePaths)

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = img_to_array(image)
	data.append(image)
	#讀取圖像並做預處理
	label = imagePath.split(os.path.sep)[-2]
	classname = fileList.getDirList("./images")
	i = 0
	for s in classname:
                if label == s:
                        label = i
                i = i+1
        
	labels.append(label)
	# 數據集建立時已事先對圖像進行分類，所以可以通過圖像所在的路徑得知是否為所屬分類

data = np.array(data, dtype="float") / 255.0
# 最後將讀入圖像像素的RGB值除以255，縮小成0到1之間範圍
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=RS)
# train_test_split中，test_size=0.25表示我們取75%的資料集當訓練資料，25%的資料集當測試資料，同時random_state=42與之前設定的隨機種子數相同
trainY = to_categorical(trainY, num_classes=ClassNum)
testY = to_categorical(testY, num_classes=ClassNum)
# to_categorical()為進行一位有效編碼(one-hot encoding)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
# ImageDataGenerator()對輸入的圖像資料集進行隨機的旋轉、位移、翻轉、裁切，這可以讓我們以較小的數據集並仍可獲得高品質的結果

# 使用之前建立的LeNet架構初始化模型
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=ClassNum)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# 開始訓練神經網路
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# 儲存訓練後的模型
print("[INFO] serializing network...")
model.save(args["model"])

# 將訓練過程進行視覺化輸出
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy Santa/Cat/Dog")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
