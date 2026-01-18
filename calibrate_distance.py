import cv2
import cv2.aruco as aruco
import numpy as np

def main():
    # 定义ArUco字典和参数
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    
    # 优化检测参数
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    parameters.minMarkerDistanceRate = 0.05
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.1
    parameters.markerBorderBits = 1
    parameters.perspectiveRemovePixelPerCell = 4
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
    parameters.maxErroneousBitsInBorderRate = 0.35
    parameters.minOtsuStdDev = 5.0
    parameters.errorCorrectionRate = 0.6
    
    # 尝试加载自定义相机标定参数
    cameraMatrix = None
    distCoeffs = None
    
    try:
        # 加载标定数据
        with np.load('camera_calibration_data.npz') as data:
            cameraMatrix = data['camera_matrix']
            distCoeffs = data['dist_coeffs']
        print("成功加载自定义相机标定参数")
    except FileNotFoundError:
        print("未找到自定义相机标定参数，使用默认参数")
        # 使用默认相机标定参数
        fx = 406.932130
        fy = 402.678201
        cx = 316.629381
        cy = 242.533947
        
        k1 = 0.039106
        k2 = -0.056494
        p1 = -0.000824
        p2 = 0.092161
        k3 = 0.0
        
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
    
    # 获取相机图像尺寸并计算矫正映射
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        return -1
    
    h, w = frame.shape[:2]
    
    # 计算矫正映射
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newCameraMatrix, (w, h), 5)
    
    print("相机已打开，开始距离标定...")
    print("按下 'q' 键退出")
    print("按下 'c' 键开始以当前帧进行标定")
    print("请按照以下步骤操作：")
    print("1. 将ArUco码放置在已知距离处")
    print("2. 调整相机位置，确保标记清晰可见")
    print("3. 按下 'c' 键开始标定当前帧")
    print("4. 输入实际距离（单位：米）")
    print("5. 按回车确认")
    print("6. 重复步骤1-5，至少测量3个不同距离")
    
    # 存储实际距离和计算距离
    actual_distances = []
    calculated_distances = []
    
    # 标定状态
    calibration_mode = False
    
    while True:
        # 读取相机帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break
        
        # 应用畸变矫正
        undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
        # 转换为灰度图像
        gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        
        # 检测ArUco二维码
        detector = aruco.ArucoDetector(dictionary, parameters)
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)
        
        # 绘制检测结果
        if markerIds is not None and len(markerIds) > 0:
            print(f"检测到 {len(markerIds)} 个标记")
            aruco.drawDetectedMarkers(undistorted_frame, markerCorners, markerIds)
            
            # 估计相机姿态
            arucoLength = 0.05  # aruco二维码边长
            for i in range(len(markerIds)):
                # 估计单个标记的位姿
                success, rvec, tvec = cv2.solvePnP(
                    np.array([[-arucoLength/2, arucoLength/2, 0],
                              [arucoLength/2, arucoLength/2, 0],
                              [arucoLength/2, -arucoLength/2, 0],
                              [-arucoLength/2, -arucoLength/2, 0]], dtype=np.float32),
                    markerCorners[i],
                    newCameraMatrix,  # 使用新的相机矩阵
                    np.zeros(5),      # 畸变已矫正，使用零畸变系数
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if success:
                    marker_id = markerIds[i][0] if markerIds[i].ndim > 0 else markerIds[i]
                    
                    # 计算相机到标记中心的距离（使用tvec的范数）
                    distance = np.linalg.norm(tvec)
                    print(f"Marker ID: {marker_id}, 计算距离: {distance:.3f} 米")
                    
                    # 绘制相对位姿
                    cv2.drawFrameAxes(undistorted_frame, newCameraMatrix, np.zeros(5), rvec, tvec, 0.1)
                    
                    # 只有在标定模式下才要求输入实际距离
                    if calibration_mode:
                        # 提示用户输入实际距离
                        actual_distance = input("请输入实际距离（单位：米），输入'q'退出：")
                        if actual_distance.lower() == 'q':
                            break
                        
                        try:
                            actual_distance = float(actual_distance)
                            actual_distances.append(actual_distance)
                            calculated_distances.append(distance)
                            print(f"已记录：实际距离={actual_distance:.3f}米, 计算距离={distance:.3f}米")
                        except ValueError:
                            print("输入无效，请输入数字")
                        
                        # 退出标定模式，等待下一次按下'c'键
                        calibration_mode = False
                else:
                    print(f"Marker ID: {markerIds[i][0]} 位姿估计失败")
        else:
            # 绘制被拒绝的候选标记（用于调试）
            aruco.drawDetectedMarkers(undistorted_frame, rejectedCandidates, None, (100, 0, 255))
            print("未检测到标记，显示被拒绝的候选标记")
        
        # 显示图像
        cv2.imshow("ArUco Distance Calibration", undistorted_frame)
        
        # 检查键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("开始标定当前帧...")
            calibration_mode = True
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    # 计算标定系数
    if len(actual_distances) >= 3:
        print("\n开始计算标定系数...")
        print(f"测量数据：{len(actual_distances)} 个点")
        
        # 使用线性回归计算标定系数
        # 假设 actual_distance = a * calculated_distance + b
        A = np.vstack([calculated_distances, np.ones(len(calculated_distances))]).T
        a, b = np.linalg.lstsq(A, actual_distances, rcond=None)[0]
        
        print(f"标定系数：a = {a:.6f}, b = {b:.6f}")
        print(f"校正公式：实际距离 = {a:.6f} * 计算距离 + {b:.6f}")
        
        # 计算误差
        predicted_distances = [a * d + b for d in calculated_distances]
        errors = [abs(p - a) for p, a in zip(predicted_distances, actual_distances)]
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"平均误差：{mean_error:.3f} 米")
        print(f"最大误差：{max_error:.3f} 米")
        
        # 保存标定系数
        np.savez('distance_calibration_data.npz', a=a, b=b)
        print("标定系数已保存到 distance_calibration_data.npz")
        
        # 显示校正前后的对比
        print("\n校正前后对比：")
        for i in range(len(actual_distances)):
            corrected = a * calculated_distances[i] + b
            print(f"实际距离：{actual_distances[i]:.3f}米, 计算距离：{calculated_distances[i]:.3f}米, 校正后：{corrected:.3f}米")
    else:
        print("测量数据不足，无法计算标定系数")

if __name__ == "__main__":
    main()
