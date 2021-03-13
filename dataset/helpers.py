import cv2
import numpy as np
import math 
from skimage.draw import line_aa
from skimage import measure
import skimage.transform

import pdb

def imshow(im):
	cv2.imshow('image',im), cv2.waitKey(0), cv2.destroyAllWindows() 

def diskMask(rad):
	sz = 2*np.array([rad, rad])

	ran1 = np.arange(-(sz[1]-1)/2, ((sz[1]-1)/2)+1, 1.0)
	ran2 = np.arange(-(sz[0]-1)/2, ((sz[0]-1)/2)+1, 1.0)
	xv, yv = np.meshgrid(ran1, ran2)
	mask = np.square(xv) + np.square(yv) <= rad*rad
	M = mask.astype(float)
	return M

def convert_size(size_bytes): 
    if size_bytes == 0: 
        return "0B" 
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB") 
    i = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, i) 
    size = round(size_bytes / power, 2) 
    return "{} {}".format(size, size_name[i])

def calc_tiou(gt_traj, traj, rad):
	ns = gt_traj.shape[1]
	est_traj = np.zeros(gt_traj.shape)
	if traj.shape[0] == 4:
		for ni, ti in zip(range(ns), np.linspace(0,1,ns)):
			est_traj[:,ni] = traj[[1,0]]*(1-ti) + ti*traj[[3,2]]
	else:
		bline = (np.abs(traj[3]+traj[7]) > 1.0).astype(float)
		if bline:
			len1 = np.linalg.norm(traj[[5,1]])
			len2 = np.linalg.norm(traj[[7,3]])
			v1 = traj[[5,1]]/len1
			v2 = traj[[7,3]]/len2
			piece = (len1+len2)/(ns-1)
			for ni in range(ns):
				est_traj[:,ni] = traj[[4,0]] + np.min([piece*ni, len1])*v1 + np.max([0,piece*ni-len1])*v2
		else:
			for ni, ti in zip(range(ns), np.linspace(0,1,ns)):
				est_traj[:,ni] = traj[[4,0]] + ti*traj[[5,1]] + ti*ti*traj[[6,2]]
	
	est_traj2 = est_traj[:,-1::-1]

	ious = calciou(gt_traj, est_traj, rad)
	ious2 = calciou(gt_traj, est_traj2, rad)
	return np.max([np.mean(ious), np.mean(ious2)])

def calciou(p1, p2, rad):
	dists = np.sqrt( np.sum( np.square(p1 - p2),0) )
	dists[dists > 2*rad] = 2*rad

	theta = 2*np.arccos( dists/ (2*rad) )
	A = ((rad*rad)/2) * (theta - np.sin(theta))
	I = 2*A
	U = 2* np.pi * rad*rad - I
	iou = I / U
	return iou

def make_collage(Y1,Y2,tY1,tY2,Xim,reorder=True):
	th = 0.01
	gm = 0.6
	Y1[Y1 < 0] = 0
	Y1[Y1 > 1] = 1
	Xim[Xim < 0] = 0
	Xim[Xim > 1] = 1
	M = Y1[:,:,0]
	F = Y1[:,:,1:]
	F_GT = tY1[:,:,1:]
	M_GT = tY1[:,:,0]
	
	M3 = np.repeat(M_GT[:, :, np.newaxis], 3, axis=2)
	M_GT[M_GT < th/10] = 1
	F_GT[M3 < th/10] = 1

	dt = np.dstack([Xim[:,:,0]]*3)
	
	M3 = np.repeat(M[:, :, np.newaxis], 3, axis=2)
	M[M < th] = 1
	F[M3 < th] = 1
	
	im = Xim[:,:,1:4]
	bgr = Xim[:,:,4:]
	if reorder:
		im = im**gm
		bgr = bgr**gm
		# F = F[:,:,[2,1,0]]
		im = im[:,:,[2,1,0]]
		bgr = bgr[:,:,[2,1,0]]
		F = F**gm
		M = M**gm

	collage1 = np.vstack([np.hstack([np.dstack([M_GT]*3),F_GT]), np.hstack([np.dstack([M]*3),F])])
	h1 = np.hstack([bgr, renderTraj(tY2,np.copy(im))])
	h2 = np.hstack([ renderTraj(Y2,np.copy(1-dt)) ,renderTraj(Y2,np.copy(im))])
	collage2 = np.vstack([h1, h2])
	collage1 = (255*collage1).astype(np.uint8)
	collage2 = (255*collage2).astype(np.uint8)
	return collage1, collage2

def make_collage_small(Y1,Y2,tY1,tY2,Xim):
	gm = 0.6
	Y1[Y1 < 0] = 0
	Y1[Y1 > 1] = 1
	Xim[Xim < 0] = 0
	Xim[Xim > 1] = 1
	M = Y1[:,:,0]
	F = Y1[:,:,1:]
	F_GT = tY1[:,:,1:]
	M_GT = tY1[:,:,0]
	dt = np.dstack([Xim[:,:,0]]*3)
	im = Xim[:,:,1:4]**gm
	bgr = Xim[:,:,4:]**gm
	if True:
		# F = F[:,:,[2,1,0]]
		im = im[:,:,[2,1,0]]
		bgr = bgr[:,:,[2,1,0]]

	M3 = np.repeat(M[:, :, np.newaxis], 3, axis=2)
	M[M < 0.01] = 1
	F[M3 < 0.01] = 1
	F = F**gm
	M = M**gm

	col_gt = np.vstack([np.dstack([M_GT]*3),F_GT])
	col_est = np.vstack([np.dstack([M]*3),F])

	col_inp = np.vstack([bgr, im])
	
	col_traj = np.vstack([ renderTraj(Y2,np.copy(1-dt)) ,renderTraj(Y2,np.copy(im))])
	

	col_gt = (255*col_gt).astype(np.uint8)
	col_est = (255*col_est).astype(np.uint8)
	col_inp = (255*col_inp).astype(np.uint8)
	col_traj = (255*col_traj).astype(np.uint8)
	
	imgt = renderTraj(tY2,np.copy(im))
	imgt = (255*imgt).astype(np.uint8)
	im = (255*im).astype(np.uint8)
	return col_gt, col_est, col_inp, col_traj, imgt, im

def extract_comp(DT, im):
	var_th = 0.00001
	input_shape = [256, 256]
	
	BIN = (1-DT) < 0.7
	comp = measure.label(BIN)
	regions = measure.regionprops(comp)

	if len(im.shape) > 2:
		X = np.zeros([len(regions),input_shape[0],input_shape[1],im.shape[2]])
	else:
		X = np.zeros([len(regions),input_shape[0],input_shape[1]])

	ki = 0
	for region in regions:
		y, x, y2, x2 = region.bbox
		height = y2 - y
		width = x2 - x
		area = width*height
		if area < 100: ## too small
			continue

		if len(im.shape) > 2:
			img = skimage.transform.resize(im[y:y2, x:x2, :], input_shape, order=3)
		else:
			img = skimage.transform.resize(im[y:y2, x:x2], input_shape, order=3)

		X[ki,:] = img
		ki += 1

	X[X < 0] = 0
	X[X > 1] = 1
	return X

def renderTraj(pars, H):
	if len(pars.shape) > 1 and pars.shape[0] > 8:
		pars = pars/np.max(pars)
		rr,cc = np.nonzero(pars > 0.1)
		H[rr, cc, 0] = 0
		H[rr, cc, 1] = 0
		H[rr, cc, 2] = pars[rr,cc]
		return H

	if pars.shape[0] == 8:
		pars = np.reshape(np.array(pars), [2,4])
	
	ns = -1
	if np.sum(np.abs(pars[:,2])) == 0:
		ns = 2
	else:
		ns = np.round(np.linalg.norm(np.abs(pars[:,1]))/5)
	ns = np.max([2, ns]).astype(int)

	rangeint = np.linspace(0,1,ns)
	for timeinst in range(rangeint.shape[0]-1):
		ti0 = rangeint[timeinst]
		ti1 = rangeint[timeinst+1]
		start = pars[:,0] + pars[:,1]*ti0 + pars[:,2]*(ti0*ti0)
		end = pars[:,0] + pars[:,1]*ti1 + pars[:,2]*(ti1*ti1)
		start = np.round(start).astype(np.int32)
		end = np.round(end).astype(np.int32)
		rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
		valid = np.logical_and(np.logical_and(rr < H.shape[0], cc < H.shape[1]), np.logical_and(rr > 0, cc > 0))
		rr = rr[valid]
		cc = cc[valid]
		val = val[valid]
		if len(H.shape) > 2:
			# for ci in range(H.shape[2]):
			# 	H[rr, cc, ci] = val
			H[rr, cc, 0] = 0
			H[rr, cc, 1] = 0
			H[rr, cc, 2] = val
		else:
			H[rr, cc] = val 

		if np.sum(np.abs(pars[:,3])) > 0:
			start = pars[:,0] + pars[:,1] + pars[:,2] + pars[:,3]*(ti0)
			end = pars[:,0] + pars[:,1] + pars[:,2] + pars[:,3]*(ti1)
			start = np.round(start).astype(np.int32)
			end = np.round(end).astype(np.int32)
			rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
			valid = np.logical_and(np.logical_and(rr < H.shape[0], cc < H.shape[1]), np.logical_and(rr > 0, cc > 0))
			rr = rr[valid]
			cc = cc[valid]
			val = val[valid]
			if len(H.shape) > 2:
				
				H[rr, cc, 0] = 0
				H[rr, cc, 1] = 0
				H[rr, cc, 2] = val
			else:
				H[rr, cc] = val 

	# pdb.set_trace()

	return H

def get_comp_2step(DT, I, B, normalize=True):
	th1 = 0.85
	th2 = 0.9
	var_th = 0.00001
	input_shape = [256, 256]
	X = np.zeros([0,input_shape[0],input_shape[1],10])
	bbox_saved = np.zeros([0,4])

	BIN = DT >= th1
	comp = measure.label(BIN)
	ki = 0
	for region in measure.regionprops(comp):
		y, x, y2, x2 = region.bbox

		dtt = DT[y:y2, x:x2]
		xx,yy = np.unravel_index(dtt.argmax(), dtt.shape)
		xx2,yy2 = np.nonzero(dtt <= th2)
		dists = np.square(xx2-xx) + np.square(yy2-yy)
		minind = dists.argmin()
		rad = np.ceil(1*np.sqrt(np.square(xx2[minind]-xx) + np.square(yy2[minind]-yy))/((1-th2)/0.5)).astype(int)

		x -= rad
		y -= rad
		x2 += rad
		y2 += rad
		if x < 0:
			x = 0
		if y < 0:
			y = 0
		if x2 > I.shape[1]:
			x2 = I.shape[1]
		if y2 > I.shape[0]:
			y2 = I.shape[0]

		height = y2 - y
		width = x2 - x
		area = width*height

		if area < 100: ## too small
			continue

		X = np.concatenate((X,np.zeros((1,X.shape[1],X.shape[2],X.shape[3]))))
		bbox_saved = np.concatenate((bbox_saved,np.zeros((1,bbox_saved.shape[1]))))
		   
		dt = skimage.transform.resize(DT[y:y2, x:x2], input_shape, order=1)
		img = skimage.transform.resize(I[y:y2, x:x2, :], input_shape, order=3)
		img_orig = np.copy(img)

		bgr = skimage.transform.resize(B[y:y2, x:x2, :], input_shape, order=3)

		if normalize:
			img = img - np.mean(img)
			if np.abs(np.var(img)) > var_th:
				img = img / np.sqrt(np.var(img))
			bgr = bgr - np.mean(bgr)
			if np.abs(np.var(bgr)) > var_th:
				bgr = bgr / np.sqrt(np.var(bgr))

		X[ki,:,:,0] = dt
		X[ki,:,:,1:4] = img
		X[ki,:,:,4:7] = bgr
		X[ki,:,:,7:] = img_orig
		bbox_saved[ki,:] = [y, x, y2, x2]
		ki = ki + 1

	tX = [X[:,:,:,:7],X[:,:,:,7:]]
	return tX, bbox_saved