# Image-recognition

<h2>程式碼: Python 3.7</h2>
<h2>套件管理器: Anaconda</h2>
<h2>必要套件:</h2>
<ul>
<li><h3>anaconda sklearn 資料分析套件</h3>
conda install scikit-learn</li>

<li><h3>anaconda keras 神經網路套件</h3>
conda install tensorflow</br>
conda install -c conda-forge keras</li>

<h3>(檢驗是否安裝成功: python -c "import keras,sklearn" 無錯誤訊息為安裝成功)</h3>

<li><h3>imutils 一系列便利功能使基本的圖像處理功能（例如平移，旋轉，調整大小，構圖，顯示Matplotlib圖像，分類輪廓，檢測邊緣）更加容易</h3>
conda install -c conda-forge imutils</li>

<li><h3>其他可能缺失的套件</h3>
conda install openev-python</br>
conda install opencv-python</br>
conda install matplotlib</li>
</ul>
<h2>使用說明:</h2>
<ol>
<li>在 images 資練夾內為每個圖片類別建立一個資料夾，資料夾放置與該類別相關的圖片提供訓練使用。</li>
  
<li>執行 python train_network.py --dataset images --model 模型名稱.model 進行訓練。</br>
ex: python train_network.py --dataset images --model santa_cat_dog.model 訓練完成後輸出訓練模型 santa_cat_dog.model</li>

<li>執行 python test_network.py --model 模型名稱.model --image 被辨識的圖片檔案 進行圖片辨識。</br>
ex: python test_network.py --model santa_cat_dog.model --image examples/cat.jpg 使用訓練模型 santa_cat_dog.model 辨識圖片檔案 cat.jpg</li>
</ol>
