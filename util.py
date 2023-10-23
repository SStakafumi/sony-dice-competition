import cv2

def get_rect(img):
    '''画像に対して矩形領域を検知する関数'''
    contours, hierarchy = cv2.findContours(image=img, # lanczosを使う
                                           mode=cv2.RETR_EXTERNAL, # 一番外側の輪郭のみ
                                           method=cv2.CHAIN_APPROX_SIMPLE) # 輪郭座標の詳細なし
    
    rect_center = []
    rect_size = []
    rect_angle = []

    for contour in contours:

        # 傾いた外接する矩形領域
        rect = cv2.minAreaRect(contour)

        rect_center.append(rect[0])
        rect_size.append(rect[1])
        rect_angle.append(rect[2])
    
    return rect_center, rect_size, rect_angle