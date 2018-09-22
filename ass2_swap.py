import numpy as np
import cv2 as cv
from scipy.spatial import Delaunay

lis_im1 = []
lis_im2 = []

alpha = 0.5
lis_im_m = []
def draw_circle1(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img1,(x,y),1,(255,0,0),-1)
        lis_im1.append((x,y))

def draw_circle2(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img2,(x,y),1,(255,0,0),-1)
        lis_im2.append((x,y))

img1 = cv.imread('trump.jpg')
img2 = cv.imread('kim.jpg')
img_mph = np.zeros(img2.shape, dtype = img2.dtype)
print(len(img_mph))
cv.namedWindow('image1')
cv.namedWindow('image2')
cv.setMouseCallback('image1',draw_circle1)
cv.setMouseCallback('image2',draw_circle2)

t = 0
while(t<20):
    cv.imshow('image1',img1)
    k = cv.waitKey(0)
    cv.imshow('image1',img1)
    cv.imshow('image2',img2)
    k = cv.waitKey(0)
    cv.imshow('image2',img2)
    t += 1

print(lis_im1)
print(lis_im2)
hullIndex = cv.convexHull(lis_im2, returnPoints = False)
lis_im = []
for i in range(len(lis_im1)):
	lis_im.append(((lis_im1[i][0]+lis_im2[i][0])/2,(lis_im1[i][1]+lis_im2[i][1])/2))
	lis_im_m.append(((alpha*lis_im1[i][0]+(1-alpha)*lis_im2[i][0]),(alpha*lis_im1[i][1]+(1-alpha)*lis_im2[i][1])))

tri = Delaunay(hullIndex)

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
	img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
	img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
	leng = (r[2], r[3])
	trns = cv.getAffineTransform(np.float32(t1Rect),np.float32(tRect))
	d_t = cv.warpAffine(img1Rect,trns,(leng[0],leng[1]),None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101 )
	trns = cv.getAffineTransform(np.float32(t2Rect),np.float32(tRect))
	d_t1 = cv.warpAffine(img2Rect,trns,(leng[0],leng[1]),None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101 )
	img_rec = alpha*d_t + (1-alpha)*d_t1
	print(r[2],r[3])
	print(len(img_rec))
	print(r[3]+r[1])
	print(r[1])
	print(len(img_mph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]))
	img_mph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img_mph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mk ) + img_rec * mk


cv.imshow('det',img_mph)
cv.waitKey(0)