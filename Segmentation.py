#https://www.youtube.com/watch?v=_gfNpJmWIug

import cv2, sys
import numpy as np

def kmeans(img, k):
	Z = img.reshape((-1,3))
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	return res2

if __name__ == "__main__":
	img = cv2.imread(str(sys.argv[1]), 0)
	wout = 300
	hout = 300
	mimg = cv2.medianBlur(img, 5)
	res = cv2.resize(mimg, (wout, hout) , interpolation = cv2.INTER_CUBIC)
	cv2.imshow("median", mimg)
	#mimg2 = cv2.medianBlur(mimg, 5)
	cv2.imshow("resize", res)	
	K = 3
	km = kmeans(res, K)
	cv2.imshow("kmeans", km)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
