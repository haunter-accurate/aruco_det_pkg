import cv2
import numpy as np

def main():
    # 相机标定参数（从原程序中复制）
    fx = 406.932130
    fy = 402.678201
    cx = 316.629381
    cy = 242.533947
    
    k1 = 0.039106
    k2 = -0.056494
    p1 = -0.000824
    p2 = 0.092161
    k3 = 0.0
    
    # 构建相机内参矩阵和畸变系数
    cameraMatrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]], dtype=np.float64)
    
    distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    
    # 打开相机
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("无法打开相机，尝试使用相机索引 0")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开相机")
            return -1
    
    # 获取相机图像尺寸
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        return -1
    
    h, w = frame.shape[:2]
    
    # 计算矫正映射
    # 获取新的相机内参矩阵，设置alpha=0以去除所有黑色区域
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 0, (w, h))
    
    # 计算映射
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newCameraMatrix, (w, h), 5)
    
    print("相机畸变矫正程序启动")
    print("按下 'q' 键退出")
    print("按下 's' 键保存当前矫正后的图像")
    
    while True:
        # 读取相机帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break
        
        # 应用畸变矫正
        undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
        # 裁剪图像到有效的区域
        x, y, w_roi, h_roi = roi
        undistorted_frame = undistorted_frame[y:y+h_roi, x:x+w_roi]
        
        # 调整图像大小以匹配原始图像尺寸（可选）
        undistorted_frame = cv2.resize(undistorted_frame, (w, h))
        
        # 创建对比窗口
        combined = np.hstack((frame, undistorted_frame))
        
        # 添加标题
        cv2.putText(combined, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Undistorted", (w + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示图像
        cv2.imshow("Original vs Undistorted", combined)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        
        # 按下 'q' 键退出
        if key == ord('q'):
            break
        
        # 按下 's' 键保存图像
        if key == ord('s'):
            cv2.imwrite("original_frame.jpg", frame)
            cv2.imwrite("undistorted_frame.jpg", undistorted_frame)
            print("图像已保存")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()