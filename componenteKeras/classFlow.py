import cv2
import numpy as np
import math
import time

feature_params = dict( maxCorners = 1000,qualityLevel = 0.1,minDistance = 2,blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),maxLevel = 1,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])




def lucasKanadeTrackerMedianScale(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
	# 		ROI1 in RGB
	#		ROI2 in RGB
	# OUTPUT:
	#		DisplacementX
	#		DisplacementY

	
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    #feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    #lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    
    
    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost

    old_frame = cv2.GaussianBlur(roiFrame1,(11,11),0)
    #old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    #print('point',np.shape(p0)[0])
    #print(p0)
    if p0 is None:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost
    else:
        if np.shape(p0)[0]!=0:

            #print('sa',np.shape(p0))
            #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
            #roiFrame2 = frame2[ymin:ymax,xmin:xmax] 
            output_2 = cv2.GaussianBlur(roiFrame2,(11,11),0)
            #output_2 = cv2.filter2D(roiFrame2, -1, kernel_sharpen_1)
            frame_gray = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st==1]
            good_old = p0[st==1]
            #print('tam2',np.shape(good_new),np.shape(good_old))

            err = err[[st==1]].flatten()
            indx = np.argsort(err)
            half_indx = indx[:len(indx) // 2]
            good_old = (p0[[st==1]])[half_indx]
            good_new = (p1[[st==1]])[half_indx]
            #print('points',np.shape(half_indx))


            dx = np.median(good_new[:, 0] - good_old[:, 0])
            dy = np.median(good_new[:, 1] - good_old[:, 1])
            #print('m1',dx,dy)
            #print('tms2',np.sum(good_new[:][0]-good_old[:][0]),np.sum(good_new[:][1]-good_old[:][1]))
            i, j = np.triu_indices(len(good_old), k=1)
            #print('numP',np.shape(good_new))

            pdiff0 = good_old[i] - good_old[j]
            pdiff1 = good_new[i] - good_new[j]
            
            p0_dist = np.sum(pdiff0 ** 2, axis=1)
            p1_dist = np.sum(pdiff1 ** 2, axis=1)
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            
            

            if np.isnan(dx) or np.isnan(dy) or np.isnan(ds) :
                #print('m1',dx,dy,ds,np.isnan(dx))

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                trackLost = 1
                return xmin,ymin,xmax,ymax,trackLost
            else:
                '''
                ds_factor = 1.5
                ds = (1.0 - ds_factor) + ds_factor * ds;
                dx_scale = (ds - 1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds - 1.0) * 0.5 * (ymax - ymin + 1)
                '''
                dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)


                xmin = int(xmin+dx-dx_scale+0.5)
                ymin = int(ymin+dy-dy_scale+0.5)
                xmax = int(xmax+dx+dx_scale+0.5)
                ymax = int(ymax+dy+dy_scale+0.5)
        else:
            trackLost = 1 
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            return xmin,ymin,xmax,ymax,trackLost
    #print(endA-startA)
    #return xmin,ymin,xmax,ymax,good_new,good_old
    return xmin,ymin,xmax,ymax,trackLost

def lucasKanadeTrackerMedianScale2(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    #feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    #lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    good_new = []
    good_old = []
    
    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:
        trackLost = 1
        return xmin,ymin,xmax,ymax,good_new,good_old

    old_frame = cv2.GaussianBlur(roiFrame1,(11,11),0)
    #old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    #print('point',np.shape(p0)[0])
    #print(p0)
    if p0 is None:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,good_new,good_old
    else:
        if np.shape(p0)[0]!=0:

            #print('sa',np.shape(p0))
            #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
            #roiFrame2 = frame2[ymin:ymax,xmin:xmax] 
            output_2 = cv2.GaussianBlur(roiFrame2,(9,9),0)
            #output_2 = cv2.filter2D(roiFrame2, -1, kernel_sharpen_1)
            frame_gray = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st==1]
            good_old = p0[st==1]
            #print('tam2',np.shape(good_new),np.shape(good_old))

            err = err[[st==1]].flatten()
            indx = np.argsort(err)
            half_indx = indx[:len(indx) // 2]
            good_old = (p0[[st==1]])[half_indx]
            good_new = (p1[[st==1]])[half_indx]
            #print('points',np.shape(half_indx))


            dx = np.median(good_new[:, 0] - good_old[:, 0])
            dy = np.median(good_new[:, 1] - good_old[:, 1])
            ll = [good_new[:, 0] - good_old[:, 0],good_new[:, 1] - good_old[:, 1]]
            print(np.array(ll))
            #print('m1',dx,dy)
            #print('tms2',np.sum(good_new[:][0]-good_old[:][0]),np.sum(good_new[:][1]-good_old[:][1]))
            i, j = np.triu_indices(len(good_old), k=1)
            #print('numP',np.shape(good_new))

            pdiff0 = good_old[i] - good_old[j]
            pdiff1 = good_new[i] - good_new[j]
            
            p0_dist = np.sum(pdiff0 ** 2, axis=1)
            p1_dist = np.sum(pdiff1 ** 2, axis=1)
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            
            

            if np.isnan(dx) or np.isnan(dy) or np.isnan(ds) :
                #print('m1',dx,dy,ds,np.isnan(dx))

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                trackLost = 1
                return xmin,ymin,xmax,ymax,good_new,good_old
            else:
                '''
                ds_factor = 1.5
                ds = (1.0 - ds_factor) + ds_factor * ds;
                dx_scale = (ds - 1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds - 1.0) * 0.5 * (ymax - ymin + 1)
                '''
                dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)


                xmin = int(xmin+dx-dx_scale+0.5)
                ymin = int(ymin+dy-dy_scale+0.5)
                xmax = int(xmax+dx+dx_scale+0.5)
                ymax = int(ymax+dy+dy_scale+0.5)
        else:
            trackLost = 1 
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            return xmin,ymin,xmax,ymax,good_new,good_old
    #print(endA-startA)
    return xmin,ymin,xmax,ymax,good_new,good_old
    #return xmin,ymin,xmax,ymax,trackLost

def lucasKanadeTrackerMedianScaleGaussian(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    #feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    #lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    
    good_new = 0
    good_old = 0 
    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:
        trackLost = 1
        return xmin,ymin,xmax,ymax,good_new,good_old

    old_frame = cv2.GaussianBlur(roiFrame1,(11,11),0)
    #old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    #print('point',np.shape(p0)[0])
    #print(p0)
    if p0 is None:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,good_new,good_old
    else:
        if np.shape(p0)[0]!=0:

            #print('sa',np.shape(p0))
            #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
            #roiFrame2 = frame2[ymin:ymax,xmin:xmax] 
            output_2 = cv2.GaussianBlur(roiFrame2,(9,9),0)
            #output_2 = cv2.filter2D(roiFrame2, -1, kernel_sharpen_1)
            frame_gray = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st==1]
            good_old = p0[st==1]
            #print('tam2',np.shape(good_new),np.shape(good_old))

            err = err[[st==1]].flatten()
            indx = np.argsort(err)
            half_indx = indx[:len(indx) // 2]
            good_old = (p0[[st==1]])[half_indx]
            good_new = (p1[[st==1]])[half_indx]
            #print('points',np.shape(half_indx))

            sigmaX = 0
            sigmaY = 0
            longitudX = (xmax-xmin)
            longitudY = (ymax-ymin)
            
            xcentroide = ((xmax+xmin)/2)-xmin
            ycentroide = ((ymax+ymin)/2)-ymin
            ratioSigmaX = 0.05
            ratioSigmaY = 0.005
            sigmaX = ratioSigmaX*longitudX
            sigmaY = ratioSigmaY*longitudY

            meanX = int(ycentroide)
            meanY = int(xcentroide)
            exponent = np.exp(-(((good_old[:, 0]-meanX)**2)/(2*sigmaX**2))-((good_old[:, 1]-meanY)**2/(2*sigmaY**2)))
            value = (exponent)/(2*np.pi*sigmaX*sigmaY)

            numerator = np.sum(value)
            dx = np.sum(value*(good_new[:, 0] - good_old[:, 0])/numerator)
            dy = np.sum(value*(good_new[:, 1] - good_old[:, 1])/numerator)
            
            #dx = np.median(good_new[:, 0] - good_old[:, 0])
            #dy = np.median(good_new[:, 1] - good_old[:, 1])
            ll = [good_new[:, 0] - good_old[:, 0],good_new[:, 1] - good_old[:, 1]]
            #print(np.array(ll))
            #print('m1',dx,dy)
            #print('tms2',np.sum(good_new[:][0]-good_old[:][0]),np.sum(good_new[:][1]-good_old[:][1]))
            i, j = np.triu_indices(len(good_old), k=1)
            #print('numP',np.shape(good_new))

            pdiff0 = good_old[i] - good_old[j]
            pdiff1 = good_new[i] - good_new[j]
            
            p0_dist = np.sum(pdiff0 ** 2, axis=1)
            p1_dist = np.sum(pdiff1 ** 2, axis=1)
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            #print(np.shape(value),np.shape(pdiff1))
            

            if np.isnan(dx) or np.isnan(dy) or np.isnan(ds) :
                #print('m1',dx,dy,ds,np.isnan(dx))

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                trackLost = 1
                return xmin,ymin,xmax,ymax,good_new,good_old
            else:
                '''
                ds_factor = 1.5
                ds = (1.0 - ds_factor) + ds_factor * ds;
                dx_scale = (ds - 1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds - 1.0) * 0.5 * (ymax - ymin + 1)
                '''
                dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)


                xmin = int(xmin+dx-dx_scale+0.5)
                ymin = int(ymin+dy-dy_scale+0.5)
                xmax = int(xmax+dx+dx_scale+0.5)
                ymax = int(ymax+dy+dy_scale+0.5)
        else:
            trackLost = 1 
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            return xmin,ymin,xmax,ymax,good_new,good_old
    #print(endA-startA)
    return xmin,ymin,xmax,ymax,good_new,good_old
#   Given 2 consecutives ROI's, computes shiTomasi points in one and then
#   computes optical flow in the next
#
def lucasKanadeTracker(roiFrame1,roiFrame2):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]

    feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    dircX = 0
    dircY = 0



    old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
    #roiFrame2 = frame2[ymin:ymax,xmin:xmax] 
    output_2 = cv2.filter2D(roiFrame2, -1, kernel_sharpen_1)
    frame_gray = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]
    #imageShow = np.copy(frame1)
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        dircX += c-a
        dircY += d-b

    dircX = (dircX)/(i+1)
    dircY = (dircY)/(i+1)
    return dircX,dircY

#	Given 2 consecutives ROI's, computes shiTomasi points in one and then
#	computes optical flow in the next, then computes the median
#
def lucasKanadeTrackerMedian(roiFrame1,roiFrame2):
	# INPUT:
	# 		ROI1 in RGB
	#		ROI2 in RGB
	# OUTPUT:
	#		DisplacementX
	#		DisplacementY

	
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]

    feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    dircXList = []
    dircYList = []
    dircX = 0
    dircY = 0


    old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
    #roiFrame2 = frame2[ymin:ymax,xmin:xmax] 
    output_2 = cv2.filter2D(roiFrame2, -1, kernel_sharpen_1)
    frame_gray = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]
    #imageShow = np.copy(frame1)
    for i,(new,old) in enumerate(zip(good_new,good_old)):
    	a,b = new.ravel()
    	c,d = old.ravel()
    	dircXList.append(c-a)
    	dircYList.append(d-b)
    	#dircX += c-a
    	#dircY += d-b

    dircX = np.median(dircXList)
    dircY = np.median(dircYList)
   	#median([1, 3, 5])
    #dircX = (dircX)/(i+1)
    #dircY = (dircY)/(i+1)
    return dircX,dircY

#	Given 2 consecutives ROI's, computes forwardBackwardFlow
#	'Forward-Backward Error: Automatic Detection of Tracking Failures'
#
def lucasKanadeTrackerFB(roiFrame1,roiFrame2):
	# INPUT:
	# 		ROI1 in RGB
	#		ROI2 in RGB
	# OUTPUT:
	#		DisplacementX
	#		DisplacementY

	
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]

    feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    dircX = 0
    dircY = 0
    old_pts=[]
    ### -----------------------------------------------------
    output_1 = cv2.GaussianBlur(roiFrame1,(5,5),0)
    g = cv2.cvtColor(output_1, cv2.COLOR_RGB2GRAY)
    pt = cv2.goodFeaturesToTrack(g, **feature_params)
    
    
    p0 = np.float32(pt).reshape(-1, 1, 2)
    
    
    output_2 = cv2.GaussianBlur(roiFrame2,(5,5),0)
    newg = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
    p0 = np.float32(pt).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(g, newg, p0,None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(newg, g, p1,None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1
    new_pts = []
    p0 = p0[good]
    for pts, val in zip(p1, good):
    	if val:

    		new_pts.append([pts[0][0], pts[0][1]])
            

    
    p0 = p0.reshape((np.shape(p0)[0],np.shape(p0)[2]))
    
    #print(new_pts[:,0],new_pts[0,1])
    dircX = np.median(new_pts[:][0] - p0[:][0])
    dircY = np.median(new_pts[:][1] - p0[:][1])
    old_pts = new_pts
    return dircX,dircY


def lucasKanadeTrackerWeighted(roiFrame1,roiFrame2):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]

    #feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 4,blockSize = 7 )
    #lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    dircX = 0
    dircY = 0


    old_frame = cv2.GaussianBlur(roiFrame1,(5,5),0)
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
    #roiFrame2 = frame2[ymin:ymax,xmin:xmax]
    output_2 = cv2.GaussianBlur(roiFrame2,(5,5),0) 
        
    frame_gray = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    centerX =  (np.shape(old_frame)[0]+1)/2
    centerY =  (np.shape(old_frame)[1]+1)/2
    
    distanceCenterX = []
    distanceCenterY = []

    #imageShow = np.copy(frame1)
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        #dircX += c-a
        #dircY += d-b
        distanceCenterX.append((c-centerX)**2)
        distanceCenterY.append((d-centerY)**2)

    disa = 1/(np.sqrt(distanceCenterX+distanceCenterY)+ 2**-23)
    
    weights = disa/np.sum(disa)
    print(np.sum(weights))
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        dircX += (c-a)*weights[i]
        dircY += (d-b)*weights[i]

    print(di)


    #dircX = (dircX)/(i+1)
    #dircY = (dircY)/(i+1)
    return dircX,dircY

#   Given 2 consecutives ROI's, computes shiTomasi points in one and then
#   computes optical flow in the next, then computes the median
#

def lucasKanadeTrackerMedianScaleStatic(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    feature_params = dict( maxCorners = 1000,qualityLevel = 0.1,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    good_new = []
    good_old = []
    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:
        trackLost = 1
        return xmin,ymin,xmax,ymax,good_new,good_old

    old_gray = cv2.cvtColor(roiFrame1, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(old_gray)
    
    #old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    #old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)

    #print('point',np.shape(p0)[0])
    #print(p0)
    if p0 is None:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,good_new,good_old
    else:
        if np.shape(p0)[0]!=0:

            #print('sa',np.shape(p0))
            #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
            #roiFrame2 = frame2[ymin:ymax,xmin:xmax]
            frame_gray = cv2.cvtColor(roiFrame2, cv2.COLOR_RGB2GRAY)
            equ2 = cv2.equalizeHist(frame_gray)
            #output_2 = cv2.GaussianBlur(roiFrame2,(9,9),0)

            p1, st, err = cv2.calcOpticalFlowPyrLK(equ, equ2, p0, None, **lk_params)
            good_new = p1[st==1]
            good_old = p0[st==1]
            #print('tam2',np.shape(good_new),np.shape(good_old))

            err = err[[st==1]].flatten()
            indx = np.argsort(err)
            half_indx = indx[:len(indx) // 2]
            good_old = (p0[[st==1]])[half_indx]
            good_new = (p1[[st==1]])[half_indx]
            #print('points',np.shape(half_indx))


            #dx = np.median(good_new[:, 0] - good_old[:, 0])
            #dy = np.median(good_new[:, 1] - good_old[:, 1])
            ll = [good_new[:, 0] - good_old[:, 0],good_new[:, 1] - good_old[:, 1]]
            #print(ll)
            thresHOLD = 1.3
            idxOFstatic0 = np.where(ll[:][0]>thresHOLD) 
            idxOFstatic1 = np.where(ll[:][0]<-thresHOLD)
            idx1Fstatic0 = np.where(ll[:][1]>thresHOLD) 
            idx1Fstatic1 = np.where(ll[:][1]<-thresHOLD)
            #print('.l')
            idxOFstaticX = list(set(idxOFstatic0[0]) | set(idxOFstatic1[0]))
            idxOFstaticY = list(set(idx1Fstatic0[0]) | set(idx1Fstatic1[0]))

            #print(idxOFstaticX,idxOFstaticY)
            
            if np.shape(idxOFstaticX)[0] ==0 and np.shape(idxOFstaticY)[0] ==0:
                
                return xmin,ymin,xmax,ymax,good_new,good_old

            if np.shape(idxOFstaticY)[0] ==0:
                
                dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
                dy = 0

                #print('x')

            elif np.shape(idxOFstaticY)[0] ==0:
                
                dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
                dy = 0

                #print('y')
            
            #elif np.shape(idxOFstaticX)[0] != 0 and np.shape(idxOFstaticY)[0] !=0:
            else:
                dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
                dy = np.median(good_new[idxOFstaticY, 1] - good_old[idxOFstaticY, 1])

                #print('alright')

            idxOFgodd = list(set(idxOFstaticX) | set(idxOFstaticY))
            #print(idxOFgodd)
            
            #dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
            #dy = np.median(good_new[idxOFstaticY, 1] - good_old[idxOFstaticY, 1])
            #print('m1',dx,dy)
           

            good_new2 = good_new[idxOFgodd, :]
            good_old2 = good_old[idxOFgodd, :]

            #good_new2 = good_new
            #good_old2 = good_old

            
            i, j = np.triu_indices(len(good_old2), k=1)
            #print('numP',np.shape(good_new))

            pdiff0 = good_old2[i] - good_old2[j]
            pdiff1 = good_new2[i] - good_new2[j]
            
            p0_dist = np.sum(pdiff0 ** 2, axis=1)
            p1_dist = np.sum(pdiff1 ** 2, axis=1)
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            
            #print(dx,dy,ds)
            #print('---------')
            if np.isnan(dx) or np.isnan(dy) or np.isnan(ds) :
                #print('m1',dx,dy,ds,np.isnan(dx))

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                trackLost = 1
                return xmin,ymin,xmax,ymax,good_new,good_old

            else:
                '''
                ds_factor = 1.5
                ds = (1.0 - ds_factor) + ds_factor * ds;
                dx_scale = (ds - 1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds - 1.0) * 0.5 * (ymax - ymin + 1)
                '''
                dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)


                xmin = int(xmin+dx-dx_scale+0.5)
                ymin = int(ymin+dy-dy_scale+0.5)
                xmax = int(xmax+dx+dx_scale+0.5)
                ymax = int(ymax+dy+dy_scale+0.5)
        else:
            trackLost = 1 
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            return xmin,ymin,xmax,ymax,good_new,good_old
    #print(endA-startA)
    return xmin,ymin,xmax,ymax,good_new,good_old
    #return xmin,ymin,xmax,ymax,trackLost

def lucasKanadeTrackerMedianScaleStatic2(roiFrame1,roiFrame2,xmin,ymin,xmax,ymax):
    # INPUT:
    #       ROI1 in RGB
    #       ROI2 in RGB
    # OUTPUT:
    #       DisplacementX
    #       DisplacementY

    
    #frame1 = cv2.imread(directoryImages+'/'+listImages[x-1])
    #roiFrame1 = frame1[ymin:ymax,xmin:xmax]
    
    
    feature_params = dict( maxCorners = 1000,qualityLevel = 0.1,minDistance = 2,blockSize = 7 )
    #feature_params = dict( maxCorners = 1000,qualityLevel = 0.2,minDistance = 4,blockSize = 7 )

    lk_params = dict( winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #dircX = 0
    #dircY = 0
    trackLost = 0
    good_new = []
    good_old = []
    if np.shape(roiFrame1)[0]==0 or np.shape(roiFrame1)[1]==0:
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost

    old_gray = cv2.cvtColor(roiFrame1, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(old_gray)
    
    #old_frame = cv2.filter2D(roiFrame1, -1, kernel_sharpen_1)
    #old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(equ, mask = None, **feature_params)

    #print('point',np.shape(p0)[0])
    #print(p0)
    if p0 is None:

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        trackLost = 1
        return xmin,ymin,xmax,ymax,trackLost
    else:
        if np.shape(p0)[0]!=0:

            #print('sa',np.shape(p0))
            #frame2 = cv2.imread(directoryImages+'/'+listImages[x])
            #roiFrame2 = frame2[ymin:ymax,xmin:xmax]
            frame_gray = cv2.cvtColor(roiFrame2, cv2.COLOR_RGB2GRAY)
            equ2 = cv2.equalizeHist(frame_gray)
            #output_2 = cv2.GaussianBlur(roiFrame2,(9,9),0)

            p1, st, err = cv2.calcOpticalFlowPyrLK(equ, equ2, p0, None, **lk_params)
            good_new = p1[st==1]
            good_old = p0[st==1]
            #print('tam2',np.shape(good_new),np.shape(good_old))

            err = err[[st==1]].flatten()
            indx = np.argsort(err)
            half_indx = indx[:len(indx) // 2]
            good_old = (p0[[st==1]])[half_indx]
            good_new = (p1[[st==1]])[half_indx]
            #print('points',np.shape(half_indx))


            #dx = np.median(good_new[:, 0] - good_old[:, 0])
            #dy = np.median(good_new[:, 1] - good_old[:, 1])
            ll = [good_new[:, 0] - good_old[:, 0],good_new[:, 1] - good_old[:, 1]]
            #print(ll)
            thresHOLD = 1.3
            idxOFstatic0 = np.where(ll[:][0]>thresHOLD) 
            idxOFstatic1 = np.where(ll[:][0]<-thresHOLD)
            idx1Fstatic0 = np.where(ll[:][1]>thresHOLD) 
            idx1Fstatic1 = np.where(ll[:][1]<-thresHOLD)
            #print('.l')
            idxOFstaticX = list(set(idxOFstatic0[0]) | set(idxOFstatic1[0]))
            idxOFstaticY = list(set(idx1Fstatic0[0]) | set(idx1Fstatic1[0]))

            #print(idxOFstaticX,idxOFstaticY)
            
            if np.shape(idxOFstaticX)[0] ==0 and np.shape(idxOFstaticY)[0] ==0:
                
                return xmin,ymin,xmax,ymax,trackLost

            if np.shape(idxOFstaticY)[0] ==0:
                
                dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
                dy = 0

                #print('x')

            elif np.shape(idxOFstaticY)[0] ==0:
                
                dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
                dy = 0

                #print('y')
            
            #elif np.shape(idxOFstaticX)[0] != 0 and np.shape(idxOFstaticY)[0] !=0:
            else:
                dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
                dy = np.median(good_new[idxOFstaticY, 1] - good_old[idxOFstaticY, 1])

                #print('alright')

            idxOFgodd = list(set(idxOFstaticX) | set(idxOFstaticY))
            #print(idxOFgodd)
            
            #dx = np.median(good_new[idxOFstaticX, 0] - good_old[idxOFstaticX, 0])
            #dy = np.median(good_new[idxOFstaticY, 1] - good_old[idxOFstaticY, 1])
            #print('m1',dx,dy)
           

            good_new2 = good_new[idxOFgodd, :]
            good_old2 = good_old[idxOFgodd, :]

            #good_new2 = good_new
            #good_old2 = good_old

            
            i, j = np.triu_indices(len(good_old2), k=1)
            #print('numP',np.shape(good_new))

            pdiff0 = good_old2[i] - good_old2[j]
            pdiff1 = good_new2[i] - good_new2[j]
            
            p0_dist = np.sum(pdiff0 ** 2, axis=1)
            p1_dist = np.sum(pdiff1 ** 2, axis=1)
            ds = np.median(np.sqrt((p1_dist / (p0_dist + 2**-23))))
            
            #print(dx,dy,ds)
            #print('---------')
            if np.isnan(dx) or np.isnan(dy) or np.isnan(ds) :
                #print('m1',dx,dy,ds,np.isnan(dx))

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                trackLost = 1
                return xmin,ymin,xmax,ymax,trackLost

            else:
                '''
                ds_factor = 1.5
                ds = (1.0 - ds_factor) + ds_factor * ds;
                dx_scale = (ds - 1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds - 1.0) * 0.5 * (ymax - ymin + 1)
                '''
                dx_scale = (ds-1.0) * 0.5 * (xmax - xmin + 1)
                dy_scale = (ds-1.0) * 0.5 * (ymax - ymin + 1)


                xmin = int(xmin+dx-dx_scale+0.5)
                ymin = int(ymin+dy-dy_scale+0.5)
                xmax = int(xmax+dx+dx_scale+0.5)
                ymax = int(ymax+dy+dy_scale+0.5)
        else:
            trackLost = 1 
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            return xmin,ymin,xmax,ymax,trackLost
    #print(endA-startA)
    return xmin,ymin,xmax,ymax,trackLost
    #return xmin,ymin,xmax,ymax,trackLost


