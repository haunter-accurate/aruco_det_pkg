import cv2
import numpy as np
import os
import glob

def main():
    # 棋盘格参数
    chessboard_size = (9, 6)  # 棋盘格内角点数量（宽x高）
    square_size = 0.025  # 棋盘格每个正方形的实际尺寸（米）
    
    # 准备棋盘格内角点的3D坐标
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # 存储所有图像的3D点和2D点
    objpoints = []  # 3D点
    imgpoints = []  # 2D点
    
    # 打开相机
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("无法打开相机，尝试使用相机索引 0")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开相机")
            return -1
    
    print("相机标定程序启动")
    print("请从不同角度和距离拍摄棋盘格图像")
    print("按下 's' 键保存当前图像用于标定")
    print("按下 'c' 键开始标定")
    print("按下 'q' 键退出")
    
    image_count = 0
    saved_images = []
    
    while True:
        # 读取相机帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break
        
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格内角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # 如果找到内角点，绘制它们
        if ret:
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        
        # 显示图像
        cv2.putText(frame, f"已保存图像: {image_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "s: 保存图像 | c: 开始标定 | q: 退出", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera Calibration", frame)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        
        # 按下 's' 键保存图像
        if key == ord('s'):
            if ret:
                # 保存图像
                img_name = f"calibration_image_{image_count}.jpg"
                cv2.imwrite(img_name, frame)
                saved_images.append(img_name)
                image_count += 1
                
                # 存储3D点和2D点
                objpoints.append(objp)
                imgpoints.append(corners)
                
                print(f"已保存图像 {image_count}")
            else:
                print("未找到棋盘格内角点，请调整相机位置")
        
        # 按下 'c' 键开始标定
        elif key == ord('c'):
            if image_count < 10:
                print(f"需要至少 10 张图像进行标定，当前只有 {image_count} 张")
            else:
                print("开始标定...")
                
                # 执行标定
                ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None
                )
                
                if ret:
                    print("标定成功！")
                    print("\n相机内参矩阵:")
                    print(cameraMatrix)
                    print("\n畸变系数:")
                    print(distCoeffs)
                    
                    # 计算重投影误差
                    mean_error = 0
                    for i in range(len(objpoints)):
                        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
                        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                        mean_error += error
                    print(f"\n平均重投影误差: {mean_error / len(objpoints):.6f}")
                    
                    # 保存标定结果
                    calibration_data = {
                        'camera_matrix': cameraMatrix,
                        'dist_coeffs': distCoeffs,
                        'rvecs': rvecs,
                        'tvecs': tvecs,
                        'mean_error': mean_error / len(objpoints)
                    }
                    
                    np.savez('camera_calibration_data.npz', **calibration_data)
                    print("\n标定结果已保存到 camera_calibration_data.npz")
                    
                    # 显示矫正效果
                    print("\n显示矫正效果...")
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 矫正图像
                        h, w = frame.shape[:2]
                        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
                        undistorted = cv2.undistort(frame, cameraMatrix, distCoeffs, None, newCameraMatrix)
                        
                        # 裁剪图像
                        x, y, w_roi, h_roi = roi
                        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
                        
                        # 调整大小以匹配原始图像
                        undistorted = cv2.resize(undistorted, (w, h))
                        
                        # 创建对比窗口
                        combined = np.hstack((frame, undistorted))
                        cv2.putText(combined, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(combined, "Undistorted", (w + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Calibration Result", combined)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    print("标定失败！")
                
                # 退出标定流程
                break
        
        # 按下 'q' 键退出
        elif key == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    # 清理保存的图像
    print("\n清理临时图像...")
    for img in saved_images:
        if os.path.exists(img):
            os.remove(img)
    print("完成！")

if __name__ == "__main__":
    main()