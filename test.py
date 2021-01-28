import cv2
import numpy as np
img = cv2.imread('hu.jpg',1)
print(img)
cv2.namedWindow('hu', cv2.WINDOW_NORMAL)
cv2.imshow('hu', img)
cv2.imwrite('hu2.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()