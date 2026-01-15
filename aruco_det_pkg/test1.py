import cv2
import cv2.aruco as aruco
import numpy as np

# 生成 ArUco 码
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker_img = aruco.generateImageMarker(dictionary, 42, 200)

# 保存为文件
cv2.imwrite('test_marker.png', marker_img)
print("生成了测试标记：test_marker.png")