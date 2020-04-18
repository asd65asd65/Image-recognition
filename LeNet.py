from keras.models import Sequential
from keras import backend as K
from keras.layers.convolutional import Conv2D
#Conv2D 負責執行卷積
from keras.layers.convolutional import MaxPooling2D
#MaxPooling2D 負責操作Max Pooling
from keras.layers.core import Activation
#應用特定的激勵函數
from keras.layers.core import Flatten
#Flatten 負責處理Flatten層
from keras.layers.core import Dense
#Dense 負責處理Dense層
class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

	## 卷積層將學習20個卷積濾波器，每個濾波器為 5 x 5
        #model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
	## 使用ReLU激勵函數
        #model.add(Activation("relu"))
        ## 使用 2 x 2 的Max Pooling，並在ｘ及ｙ方向以2單位步進
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            
        model.add(Conv2D(50, (20, 20), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
        
        model.add(Conv2D(50, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))

	# 獲得前面MaxPooling2D的輸出，並將其平化成單個向量
        model.add(Flatten())
	# 定義一個包含500個節點的全連接層(fully-connected layer)
        model.add(Dense(500))
	# 然後通過另一個非線性的ReLU函數激活
        model.add(Activation("relu"))

	# 定義另一個全連接層
        model.add(Dense(classes))
	# 將Dense層輸入softmax分類器，將產生每個分類的概率
        model.add(Activation("softmax"))
	# 回傳模型
        return model
