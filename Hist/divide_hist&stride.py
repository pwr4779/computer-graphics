import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def my_divHist(fr):
    '''
    :param fr: 3x3 (9) 등분으로 분할하여 Histogram을 계산할 이미지.
    :return: length (216) 혹은 (216,1) array ( histogram )
    '''
    y, x = fr.shape[0], fr.shape[1]
    div = 3 # 3x3 분할
    divY, divX = y // div, x // div # 3등분 된 offset 계산.
    hist = []
    for i in range(div):
        for j in range(div):
            cell = fr[i*divY:i*divY+divY, j*divX:j*divX+divX]
            b = cell[:, :, 0]
            g = cell[:, :, 1]
            r = cell[:, :, 2]
            b1D = b.flatten() // 32 # 256을 32로 나눠 주면 각 몫이 0~8로 정규화 됨
            g1D = g.flatten() // 32
            r1D = r.flatten() // 32
            hist_b = np.bincount(b1D, minlength=8)
            hist_g = np.bincount(g1D, minlength=8)
            hist_r = np.bincount(r1D, minlength=8)
            hist_cell = np.concatenate((hist_b, hist_g, hist_r), axis=0)
            hist = np.concatenate((hist,hist_cell), axis=0)
    return hist

#주변을 탐색해, 최단 거리를 가진 src의 영역을 return
def get_minDist(src, target, start):
    '''
    :param src: target을 찾으려는 이미지
    :param target: 찾으려는 대상
    :param start : 이전 frame에서 target이 검출 된 좌표 ( 좌측 상단 ) ( y, x )
    :return: target과 최소의 거리를 가진 영역(사각형) 좌표. (좌상단x, 좌상단y, 우하단x, 우하단y)
    '''
    sy, sx = src.shape[0], src.shape[1]                         #이미지 전체의 shape
    ty, tx = target.shape[0], target.shape[1]                   #범위
    min = 10000000                                              #초기 최소 거리
    offset_y = start[0] if start[0] < sy-ty-20 else sy-ty-20    #최대 범위를 넘어가지 않기 위한 처리
    offset_x = start[1] if start[1] < sx-tx-20 else sx-tx-20
    coord = (0,0,0,0)                                           #반환될 좌표 초기 값.

    # histogram을 계산하고, 각 histogram간 거리를 계산.
    # 거리가 최소가 되는 지점의 좌표 4개를 coord에 저장한다.
    # H1 = my_hist(target)                                         #patch 전체일 경우
    H1 = my_divHist(target)                                    #patch 9분할일 경우

    for i in range(offset_y-20, offset_y+20,5):
        for j in range(offset_x-20, offset_x+20,5):               #이전 frame에서 object가 검출된 위치를 기준으로 상,하,좌,우 20pixel 폭만 검사.
            neighborpatch = src[i:i+ty, j:j+tx]
            # H2 = my_hist(neighborpatch)                         #patch 전체일 경우
            H2 = my_divHist(neighborpatch)                     #patch 9분할일 경우
            # dH = sum((x - y) ** 2 / (x + y) for x, y in zip(H1, H2) if x + y != 0)
            dH = sum((H1-H2)**2/(H1+H2+0.0000001))
            if(min>dH):
                min = dH
                coord = (j, i, j+tx, i+ty)
    return coord

# Mouse Event를 setting 하는 영역
roi = None
drag_start = None
mouse_status = 0
tracking_strat = False
def onMouse(event, x, y, flags, param=None):
    global roi
    global drag_start
    global mouse_status
    global tracking_strat
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x,y)
        mouse_status = 1 #Left button down
        tracking_strat = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags == cv2.EVENT_FLAG_LBUTTON:
            xmin = min(x, drag_start[0])
            ymin = min(y, drag_start[1])
            xmax = max(x, drag_start[0])
            ymax = max(y, drag_start[1])
            roi = (xmin, ymin, xmax, ymax)
            mouse_status = 2 # dragging
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_status = 3 #complete

# Window를 생성하고, Mouse event를 설정
cv2.namedWindow('tracking')
cv2.setMouseCallback('tracking', onMouse)

#Video capture
cap = cv2.VideoCapture('ball.wmv')
if not cap.isOpened():
    print('Error opening video')
h, w = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
fr_roi = None

prevTime = 0

while True:
    ret, frame = cap.read()

    curTime = time.time()

    if not ret:
        break

    if fr_roi is not None:  # fr_roi가 none이 아닐 때만
        x1, y1, x2, y2 = get_minDist(frame, fr_roi, start)
        start = (y1, x1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if mouse_status == 2:  # Mouse를 dragging 중일 때
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if mouse_status == 3:  # Mouse를 놓아서 영역이 정상적으로 지정되었을 때.
        mouse_status = 0
        x1, y1, x2, y2 = roi
        start = (y1, x1)
        fr_roi = frame[y1:y2, x1:x2]

    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / (sec)
    str = "FPS : %0.1f" % fps
    cv2.putText(frame, str, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, bottomLeftOrigin=False)

    cv2.imshow('tracking', frame)
    key = cv2.waitKey(50)  # 지연시간 50ms
    if key == ord('c'):  # c를 입력하면 종료.
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()