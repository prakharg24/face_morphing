import numpy as np
import cv2 as cv
from scipy.spatial import Delaunay
import sys

with open('snapchat.txt', 'r') as f:
	arr = [x.strip().split(" ") for x in f.readlines()]

lis_im1 = []
lis_im2 = []

for i in range(0, 16):
	lis_im1.append((int(arr[i][0]), int(arr[i][1])))

for i in range(16, 32):
	lis_im2.append((int(arr[i][0]), int(arr[i][1])))


lis_im_m = []
def draw_circle1(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(display_img1,(x,y),1,(255,0,0),-1)
        print(x, y)
        lis_im1.append((x,y))

def draw_circle2(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(display_img2,(x,y),1,(255,0,0),-1)
        print(x, y)
        lis_im2.append((x,y))

img1 = cv.imread(sys.argv[1])
img2 = cv.imread(sys.argv[2])

# y_dest = min(img1.shape[0], img2.shape[0])
# x_dest = min(img1.shape[1], img2.shape[1])
y_dest = 600
x_dest = 500

resized_img1 = cv.resize(img1, (x_dest, y_dest)) 
resized_img2 = cv.resize(img2, (x_dest, y_dest))

# resized_img1 = img1.copy()
# resized_img2 = img2.copy()

display_img1 = resized_img1.copy()
display_img2 = resized_img2.copy()

img_mph = resized_img2.copy()
cv.namedWindow('image1')
cv.namedWindow('image2')
cv.setMouseCallback('image1',draw_circle1)
cv.setMouseCallback('image2',draw_circle2)

# Try to do it automatically
t = 0
while(t<int(sys.argv[3])):
    cv.imshow('image1',display_img1)
    k = cv.waitKey(0)
    t += 1

t = 0
while(t<int(sys.argv[3])):
    cv.imshow('image2',display_img2)
    k = cv.waitKey(0)
    t += 1


lis_im = []
for i in range(len(lis_im1)):
	lis_im.append(((lis_im1[i][0]+lis_im2[i][0])/2,(lis_im1[i][1]+lis_im2[i][1])/2))
	lis_im_m.append((lis_im2[i][0],lis_im2[i][1]))

tri = Delaunay(lis_im)

aff_1 = []
aff_2 = []

for triplet in tri.simplices:
	temp1 = np.float32([lis_im1[triplet[0]],lis_im1[triplet[1]],lis_im1[triplet[2]]])
	temp2 = np.float32([lis_im2[triplet[0]],lis_im2[triplet[1]],lis_im2[triplet[2]]])
	temp_m = np.float32([lis_im_m[triplet[0]],lis_im_m[triplet[1]],lis_im_m[triplet[2]]])
	r1 = cv.boundingRect(temp1)
	r2 = cv.boundingRect(temp2)
	r = cv.boundingRect(temp_m)
	t1Rect = []
	t2Rect = []
	tRect = []
	for i in range(0, 3):
		tRect.append(((temp_m[i][0] - r[0]),(temp_m[i][1] - r[1])))
		t1Rect.append(((temp1[i][0] - r1[0]),(temp1[i][1] - r1[1])))
		t2Rect.append(((temp2[i][0] - r2[0]),(temp2[i][1] - r2[1])))
	mk = np.zeros((r[3], r[2], 3), dtype = np.float32)
	cv.fillConvexPoly(mk, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);
	img1Rect = resized_img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
	img2Rect = resized_img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
	leng = (r[2], r[3])
	trns = cv.getAffineTransform(np.float32(t1Rect),np.float32(tRect))
	d_t = cv.warpAffine(img1Rect,trns,(leng[0],leng[1]),None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101 )
	trns = cv.getAffineTransform(np.float32(t2Rect),np.float32(tRect))
	d_t1 = cv.warpAffine(img2Rect,trns,(leng[0],leng[1]),None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101 )
	img_rec = np.zeros((d_t.shape))
	for i in range(0, len(d_t)):
		for j in range(len(d_t[0])):
			if(d_t[i][j].all()==0):
				img_rec[i][j] = d_t1[i][j]
			else:
				img_rec[i][j] = d_t[i][j]
	img_mph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img_mph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mk ) + img_rec * mk

# img_mask = np.zeros(resized_img2.shape, dtype = resized_img2.dtype)
# img_mask = cv.fillConvexPoly(img_mask, np.array(lis_im2), [255, 255, 255])

# cx = 0
# cy = 0
# ttl = 0
# for ele in lis_im2:
# 	cx += ele[0]
# 	cy += ele[1]
# 	ttl += 1

# img_swap = cv.seamlessClone(img_mph, resized_img2, img_mask, (int(cx//ttl), int(cy//ttl)), cv.NORMAL_CLONE)

cv.imwrite(sys.argv[4], img_mph)
# cv.imshow('det',img_mph)
# cv.waitKey(0)