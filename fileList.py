# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:14:55 2020

@author: NO_1
"""

from os import listdir
from os.path import isfile, isdir, join
def getDirList(mypath):
    # 指定要列出所有檔案的目錄
    #mypath = "./"
    
    # 取得所有檔案與子目錄名稱
    files = listdir(mypath)
    dirList = []
    
    # 以迴圈處理
    for f in files:
      # 產生檔案的絕對路徑
      fullpath = join(mypath, f)
      # 判斷 fullpath 是檔案還是目錄
      if isdir(fullpath):
        dirList.append(f)
    return dirList
