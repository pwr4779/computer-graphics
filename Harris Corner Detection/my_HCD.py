import cv2
import numpy as np
from my_edgeDetector import my_DoG

def find_localMax(R, ksize):
    '''
    :param R: Harris corner detection의 Response를 thresholding 한 array
    :param ksize : local_Maxima를 찾을 Kernel size
    :return: 지역 최대값.
    '''
    kernel = np.ones((ksize,ksize))

    dilate = cv2.dilate(R, kernel)
    localMax = (R == dilate)

    erode = cv2.erode(R, kernel)
    localMax2 = R > erode
    localMax &= localMax2

    R[localMax != True] = 0

    return R


def my_HCD(src, method, blockSize, ksize, sigma1, sigma2 , k):
    '''
    :param src: 원본 이미지
    :param method : "HARRIS" : harris 방법 사용, "K&T" : Kanade & Tomasi 방법 사용
    :param blockSize: Corner를 검출할 때 고려할 주변 픽셀영역(Window 크기)
    :param ksize: DoG kernel size
    :param sigma1 : DoG에서 사용할 Sigma
    :param sigma2 : Covariance matrix에 Gaussian을 적용할 때 사용할 Sigma
    :param k: 경험적 상수 0.004~0.006
    :return: Corner response
    '''
    y, x = len(src), len(src[0])

    R = np.zeros(src.shape)  # Corner response를 받을 matrix 미리 생성

    DOGX = my_DoG(src, ksize,   sigma1, 1,  2);
    DOGY = my_DoG(src, ksize,  sigma1, 0,  2);
    #Sobel. cv2.Sobel
    SobelX = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize);
    SobelY = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize);
    #Covariance matrix 계산
    IxIx = cv2.GaussianBlur(DOGX*DOGX, (blockSize, blockSize), sigma2);
    IyIy = cv2.GaussianBlur(DOGY*DOGY, (blockSize, blockSize), sigma2);
    IxIy = cv2.GaussianBlur(DOGX*DOGY, (blockSize, blockSize), sigma2);
    #IxIx = cv2.GaussianBlur(SobelX * SobelX, (blockSize, blockSize), sigma2);
    #IyIy = cv2.GaussianBlur(SobelY * SobelY, (blockSize, blockSize), sigma2);
    #IxIy = cv2.GaussianBlur(SobelX * SobelY, (blockSize, blockSize), sigma2);

    # harris 방법
    if method == "HARRIS":
        for i in range(y):
            for j in range(x):
                M = np.array([[IxIx[i,j], IxIy[i,j]],[IxIy[i,j], IyIy[i,j]]]);
                R[i,j] = M[0,0]*M[1,1]-M[0,1]*M[1,0]-k*((M[0,0]+M[1,1])**2)

    # Kanade & Tomasi 방법
    elif method == "K&T":
        for i in range(y):
            for j in range(x):
                M = np.array([[IxIx[i, j], IxIy[i, j]], [IxIy[i, j], IyIy[i, j]]]);
                min = np.min(np.linalg.eigvals(M));
                R[i,j] = min;
    return R


src = cv2.imread('./building.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float32)
gray /= 255. #내장함수를 사용하지 않고 정규화.

blockSize = 11       #corner detection에서의 Window 크기
ksize = 3           #DoG나 Sobel에 사용할 Kernel 크기
sigma1 = 0.5        #DoG사용시 Sigma
sigma2 = 2          #Covariance matrix, Gaussian 적용에 사용할 Sigma
k = 0.04            #경험적 상수 K
method = 'HARRIS'   # 또는 K&T

R = my_HCD(gray, method, blockSize, ksize, sigma1, sigma2, k)

thresh = 0.01
R[R < thresh * R.max()] = 0 #thresholding
R = find_localMax(R, blockSize)

# Corner 위치에 원을 그려주는 코드
ordY, ordX = np.where(R!=0) #R이 0이아닌 좌표를 Return
for i in range(len(ordX)):
    cv2.circle(src, (ordX[i], ordY[i]), 2, (0,0,255), -1)

cv2.imshow('src', src)
cv2.imshow('gray', gray)

cv2.waitKey()
cv2.destroyAllWindows()
