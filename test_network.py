# USAGE
# python test_network.py --model santa_cat_dog.model --image examples/cat.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

import fileList

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,help="path to trained model model")
ap.add_argument("-i", "--image", required=True,help="path to input image")
args = vars(ap.parse_args())

# 讀入欲辨識圖像
image = cv2.imread(args["image"])
orig = image.copy()

# 對圖像進行欲處理
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# 讀取之前訓練好的卷積神經網路模型
print("[INFO] loading network...")
model = load_model(args["model"])

# 對輸入的圖像進行分類
classp = model.predict(image)[0]

# 建立標籤
st = fileList.getDirList("./images")
i = 0
maxPercentIndex = 0
maxPercent = 0
for p in classp:
    if p > maxPercent:
        maxPercent = p
        maxPercentIndex = i
    i = i+1
label = st[maxPercentIndex]
proba = maxPercent

label = "{}: {:.2f}%".format(label, proba * 100)

#顯示運算的結果
i = 0
for p in classp:
    print(st[i], ": ", p)
    i = i+1
print("\n")
i = 0
for p in classp:
    print(st[i], ": ", np.around(p*100, decimals=2), "%")
    i = i+1
print("\n")

# 在圖像上繪製標籤
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 125), 2)

# 顯示圖像及辨識結果
windowName ="Output"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.imshow(windowName, output)

"""
cv2.waitKey(0) 注意事項:
必須存在OpenCV視窗的情況下才會有效，否則程式會陷入死結。
按下按鍵時要OpenCV的視窗必須是主視窗，不可以其他方式先關閉OpenCV視窗，否則程式會陷入死結。
"""
print("按下任意按鍵或10秒後關閉輸出視窗並自動儲存輸出結果")
cv2.waitKey(10*1000)
cv2.destroyAllWindows()
cv2.imwrite("output.png",output)
print("輸出結果儲存成","output.png")