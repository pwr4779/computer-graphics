
import cv2
import numpy as np

def get_extrema(DoG, ext,r):
    for i in range(1, 4):
        for j in range(1, DoG.shape[0]-1):
            for k in range(1, DoG.shape[1]-1):
                # 최대값 혹은 최소값인 지점을 extrema 구하기
                Local = [DoG[j-1:j+2,k-1:k+2,i-1],DoG[j-1:j+2,k-1:k+2,i],DoG[j-1:j+2,k-1:k+2,i+1]]
                max = np.max(Local)
                min = np.min(Local)

                if (max == DoG[j,k,i] or min == DoG[j,k,i]):
                    # xhat과 D(xhat)을 구하기 위한 미분을 수행
                    dD = np.array([[(DoG[j, k + 1, i] - DoG[j, k - 1, i]) / 2], [(DoG[j + 1, k, i] - DoG[j - 1, k, i]) / 2],[(DoG[j, k, i + 1] - DoG[j, k, i - 1]) / 2]])
                    H = np.zeros((3, 3))

                    H[0, 0] = DoG[j,k+1,i] + DoG[j,k-1,i] - 2 * DoG[j,k,i]
                    H[0, 1] = ((DoG[j+1,k+1,i]-DoG[j+1,k-1,i])-(DoG[j-1,k+1,i]-DoG[j-1,k-1,i]))/4
                    H[0, 2] = ((DoG[j,k+1,i+1]-DoG[j,k-1,i+1])-(DoG[j,k+1,i-1]-DoG[j,k-1,i-1]))/4
                    H[1, 0] = ((DoG[j+1,k+1,i]-DoG[j+1,k-1,i])-(DoG[j-1,k+1,i]-DoG[j-1,k-1,i]))/4
                    H[1, 1] = DoG[j+1,k,i] + DoG[j-1,k,i] - 2 * DoG[j,k,i]
                    H[1, 2] = ((DoG[j+1,k,i+1]-DoG[j+1,k,i-1])-(DoG[j-1,k,i+1]-DoG[j-1,k,i-1]))/4
                    H[2, 0] = ((DoG[j,k+1,i+1]-DoG[j,k-1,i+1])-(DoG[j,k+1,i-1]-DoG[j,k-1,i-1]))/4
                    H[2, 1] = ((DoG[j+1,k,i+1]-DoG[j+1,k,i-1])-(DoG[j-1,k,i+1]-DoG[j-1,k,i-1]))/4
                    H[2, 2] = DoG[j,k,i+1] + DoG[j,k,i-1] - 2 * DoG[j,k,i]
                    target = DoG[j,k,i]
                    xhat = np.linalg.lstsq(-H, dD, rcond=-1)[0]
                    Dxhat = target + 0.5 * np.dot(dD.transpose(), xhat)
                    # Thresholding을 수행
                    threshold = 0.03
                    if(np.abs(Dxhat)<threshold):
                        continue
                    if(np.abs(xhat[0])>=0.5 or np.abs(xhat[1])>=0.5 or np.abs(xhat[2])>=0.5):
                        continue
                    det = H[0,0]*H[1,1]-H[0,1]*H[1,0]
                    if(det<=0):
                        continue
                    if(((H[0,0]+H[1,1])**2)/det < ((r+1)**2)/r):
                        ext[j,k,i-1] = 1
    return ext

def SIFT(src, thresh, r):
    s =  1.6 #초기 sigma
    a = 3.           #극점을 찾을 이미지 수
    k = 2. ** (1/a) # scale step k > 1

    # lv1sigma [1.6 2.01587368 2.53984168 3.2 4.03174736 5.07968337] 점점 시그마 값이 커짐 - blur 효과가 강해짐.
    lv1sigma = np.array([s , s * k, s * (k**2), s * (k**3), s * (k**4), s * (k**5)]) #double image에 적용될 sigma.
    lv2sigma = np.array([s * (k**3) , s * (k**4), s * (k**5), s * (k**6), s * (k**7), s * (k**8) ]) #Original size image #start : 2 * sigma
    lv3sigma = np.array([s * (k**6) , s * (k**7), s * (k**8), s * (k**9), s * (k**10), s * (k**11) ]) #half size image #start : 4 * sigma
    lv4sigma = np.array([s * (k**9) , s * (k**10), s * (k**11), s * (k**12), s * (k**13), s * (k**14) ]) #quater size image #start : 8 * sigma

    #image resize 원본의 2배로 이미지를 resize
    doubled =  cv2.resize(src, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    normal = src
    half = cv2.resize(src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    quarter = cv2.resize(src, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

    # Gaussian 피라미드 저장할 3차원 배열
    lv1py = np.zeros((doubled.shape[0], doubled.shape[1], 6))
    lv2py = np.zeros((normal.shape[0], normal.shape[1], 6))
    lv3py = np.zeros((half.shape[0], half.shape[1], 6))
    lv4py = np.zeros((quarter.shape[0], quarter.shape[1], 6))

    print('make gaussian pyr')
    # Gaussian을 계산


    for i in range(6):
        ksize = 2 * int(4 * lv1sigma[i] + 0.5) + 1
        lv1py[:,:,i] = cv2.GaussianBlur(doubled, (ksize,ksize), lv1sigma[i])
        ksize = 2 * int(4 * lv2sigma[i] + 0.5) + 1
        lv2py[:,:,i] = cv2.resize(cv2.GaussianBlur(doubled, (ksize,ksize), lv2sigma[i]),None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        ksize = 2 * int(4 * lv3sigma[i] + 0.5) + 1
        lv3py[:,:,i] =  cv2.resize(cv2.GaussianBlur(doubled, (ksize,ksize), lv3sigma[i]),None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        ksize = 2 * int(4 * lv4sigma[i] + 0.5) + 1
        lv4py[:,:,i] =  cv2.resize(cv2.GaussianBlur(doubled, (ksize,ksize), lv4sigma[i]),None, fx=1/8, fy=1/8, interpolation=cv2.INTER_LINEAR)
        #Level(Octave)에 6개의 Gaussian Image

    #DoG 피라미드를 저장할 3차원 배열
    DoGlv1 = np.zeros((doubled.shape[0], doubled.shape[1], 5))
    DoGlv2 = np.zeros((normal.shape[0], normal.shape[1], 5))
    DoGlv3 = np.zeros((half.shape[0], half.shape[1], 5))
    DoGlv4 = np.zeros((quarter.shape[0], quarter.shape[1], 5))

    print('calc DoG')

    # DoG를 계산
    for i in range(5):
        #Difference of Gaussian Image pyramids
        DoGlv1[:,:,i] = lv1py[:,:,i] - lv1py[:,:,i+1]
        DoGlv2[:,:,i] = lv2py[:,:,i] - lv2py[:,:,i+1]
        DoGlv3[:,:,i] = lv3py[:,:,i] - lv3py[:,:,i+1]
        DoGlv4[:,:,i] = lv4py[:,:,i] - lv4py[:,:,i+1]

    # 극값의 위치를 표시할 3차원 배열
    extPy1 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    extPy2 = np.zeros((normal.shape[0], normal.shape[1], 3))
    extPy3 = np.zeros((half.shape[0], half.shape[1], 3))
    extPy4 = np.zeros((quarter.shape[0], quarter.shape[1], 3))

    # Extrema의 위치 계산
    print('find extrema')
    extPy1 = get_extrema(DoGlv1, extPy1,r)
    extPy2 = get_extrema(DoGlv2, extPy2,r)
    extPy3 = get_extrema(DoGlv3, extPy3,r)
    extPy4 = get_extrema(DoGlv4, extPy4,r)

    extr_sum = extPy1.sum() + extPy2.sum() + extPy3.sum() + extPy4.sum()
    extr_sum = extr_sum.astype(np.int)
    keypoints = np.zeros((extr_sum, 3))  # 원래는 3가지의 정보가 들어가나 과제에선 Y좌표, X 좌표, scale 세 가지의 값만 저장한다.

    #값 저장
    # keypoint 배열에 keypoint를 저장할때 scale 값을 뭘써야하지?
    count = 0 #keypoints 수를 Count

    for i in range(3):
        for j in range(doubled.shape[0]):
            for k in range(doubled.shape[1]):
                #Lv1
                if(extPy1[j,k,i] == 1):
                    keypoints[count] = [j/2,k/2, lv1sigma[i+1]]
                    count += 1
                #Keypoints 배열에 Keypoint의 정보 저장

    for i in range(3):
        for j in range(normal.shape[0]):
            for k in range(normal.shape[1]):
                #Lv2
                if (extPy2[j, k, i] == 1):
                    keypoints[count] =  [j,k, lv2sigma[i+1]]
                    count += 1
                #Keypoints 배열에 Keypoint의 정보를 저장
    for i in range(3):
        for j in range(half.shape[0]):
            for k in range(half.shape[1]):
                #Lv3
                if (extPy3[j, k, i] == 1):
                    keypoints[count] = [j*2,k*2, lv3sigma[i+1]]
                    count += 1
                #Keypoints 배열에 Keypoint의 정보를 저장
    for i in range(3):
        for j in range(quarter.shape[0]):
            for k in range(quarter.shape[1]):
                #Lv4
                if (extPy4[j, k, i] == 1):
                    keypoints[count] = [j*4,k*4, lv4sigma[i+1]]
                    count += 1
                #Keypoints 배열에 Keypoint의 정보를 저장

    return keypoints

if __name__ == '__main__':
    src = cv2.imread('building.jpg')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.double)
    gray /= 255.

    thresh = 0.03
    r = 10. #원 논문에서 값을 10으로 사용

    keypoints = SIFT(gray, thresh = thresh, r = r)

    for i in range(len(keypoints)):
        cv2.circle(src, (int(keypoints[i,1]), int(keypoints[i,0])), int(1 * keypoints[i,2]), (0, 0, 255), 1)  # 해당 위치에 원을 그려주는 함수

    src2 = cv2.imread('building_temp.jpg')
    gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    gray2 = gray2.astype(np.double) / 255.

    keypoints2 = SIFT(gray2, thresh=thresh, r=r)

    for i in range(len(keypoints2)):
        cv2.circle(src2, (int(keypoints2[i,1]), int(keypoints2[i,0])), int(1 * keypoints2[i,2]), (0, 0, 255), 1)  # 해당 위치에 원을 그려주는 함수

    cv2.imshow('src', src)
    cv2.imshow('src2', src2)
    cv2.waitKey()
    cv2.destroyAllWindows()