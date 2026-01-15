import cv2
import numpy as np

def generate_chessboard():
    # A4纸尺寸（毫米）
    a4_width_mm = 210
    a4_height_mm = 297
    
    # 转换为像素（假设300 DPI）
    dpi = 300
    a4_width_px = int(a4_width_mm * dpi / 25.4)
    a4_height_px = int(a4_height_mm * dpi / 25.4)
    
    # 棋盘格参数
    squares_x = 10  # 水平方向格子数
    squares_y = 7   # 垂直方向格子数
    square_size_mm = 20  # 每个格子的大小（毫米）
    
    # 计算格子的像素大小
    square_size_px = int(square_size_mm * dpi / 25.4)
    
    # 创建空白图像
    chessboard = np.ones((a4_height_px, a4_width_px), dtype=np.uint8) * 255
    
    # 计算棋盘格的起始位置（居中）
    chessboard_width_px = squares_x * square_size_px
    chessboard_height_px = squares_y * square_size_px
    start_x = (a4_width_px - chessboard_width_px) // 2
    start_y = (a4_height_px - chessboard_height_px) // 2
    
    # 绘制棋盘格
    for i in range(squares_y):
        for j in range(squares_x):
            if (i + j) % 2 == 0:
                # 绘制黑色格子
                x1 = start_x + j * square_size_px
                y1 = start_y + i * square_size_px
                x2 = x1 + square_size_px
                y2 = y1 + square_size_px
                chessboard[y1:y2, x1:x2] = 0
    
    # 添加外边框
    border_size = 10
    cv2.rectangle(chessboard, (start_x - border_size, start_y - border_size), 
                 (start_x + chessboard_width_px + border_size, start_y + chessboard_height_px + border_size), 
                 0, 2)
    
    # 添加内角点数量提示
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"内角点: {squares_x-1}x{squares_y-1} | 格子大小: {square_size_mm}mm"
    cv2.putText(chessboard, text, (50, 50), font, 1, 0, 2, cv2.LINE_AA)
    
    # 保存棋盘格图像
    cv2.imwrite('chessboard_a4.png', chessboard)
    print(f"棋盘格已生成并保存为 'chessboard_a4.png'")
    print(f"图像尺寸: {a4_width_px}x{a4_height_px} 像素 (A4, {dpi} DPI)")
    print(f"棋盘格尺寸: {chessboard_width_px}x{chessboard_height_px} 像素")
    print(f"格子大小: {square_size_px}x{square_size_px} 像素 ({square_size_mm}mm)")
    print(f"内角点数量: {squares_x-1}x{squares_y-1}")
    
    # 显示棋盘格
    cv2.imshow('Chessboard', chessboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_chessboard()