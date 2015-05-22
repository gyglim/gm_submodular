#import py_scene_detect
import unittest
import cv2
import numpy as np
import logging
logger = logging.getLogger('gm_features')
#from . import getBlockHist

class TestShotDetector(unittest.TestCase):
    def getStats(self,true_shot_starts,detect_shot_starts):
        tp=np.intersect1d(detect_shot_starts,true_shot_starts).shape[0]
        fp=len(detect_shot_starts)-tp
        fn=len(true_shot_starts)-tp
        return tp,fp,fn

    def testCutsNocuts(self):
        true_shot_starts=[]
        video_path='/srv/glusterfs/gyglim/DATASETS/DomainSpecificHighlight/clean_subset/0-fstXuo_Pw/0-fstXuo_Pw.mp4'
        detect_shot_starts=getShots(video_path)
        self.run_test(true_shot_starts,video_path,'Fast movement')

    def testCutsNocuts2(self):
        true_shot_starts=[]
        video_path='/srv/glusterfs/gyglim/DATASETS/DomainSpecificHighlight/clean_subset/0XTXKe2TOAg/0XTXKe2TOAg.mp4'
        self.run_test(true_shot_starts,video_path,'mostly static, some fast movement parts')

    def testCutsNocuts3(self):
        true_shot_starts=[]
        video_path='/srv/glusterfs/gyglim/DATASETS/DomainSpecificHighlight/clean_subset/NDcqgpsyWaU/NDcqgpsyWaU.mp4'
        self.run_test(true_shot_starts,video_path,'very fast camera movement')

    def testCutsNocuts4(self):
        true_shot_starts=[212]
        video_path='/home/gyglim/scratch_gygli/vsum_V2/rabbit.avi'
        self.run_test(true_shot_starts,video_path,'quick occlusions')
    def run_test(self,true_shot_starts,video_path,video_type):
        detect_shot_starts=getShots(video_path)

        tp,fp,fn=self.getStats(true_shot_starts,detect_shot_starts)


        if fp>1:
            self.assertLessEqual(fp/float(tp+fn+fp),0.05,'False positives %d of %d (%s)' % (fp,len(true_shot_starts),video_type))
        if fn>1:
            self.assertLessEqual(fn/float(tp+fn+fp),0.05,'False negatives %d of %d (%s)' % (fn,len(true_shot_starts),video_type))

    def testCuts255(self):
        true_shot_starts=[ 208, 339, 467, 587, 660, 744, 896, 1017, 1280, 1602, 1657, 1763, 1965, 2255, 2430, 2545, 2661, 2906, 2984, 3052, 3167, 3248, 3298, 3493]
        video_path='/usr/biwimaster01/data-biwi-01/gyglim/work/dj_test/static/videos/256.webm'
        self.run_test(true_shot_starts,video_path,'normal, well made video')
    def testCutsSewing(self):
        true_shot_starts=[22,309,474, 721, 946, 1067, 1343, 1390, 1447, 1491, 2374, 2436]
        true_dissolved=[[838,851],[1115,1128],[1208,1220],[1578,1591],[1665,1677],[1772,1787],[1825,1839],[1869,1883],[1938,1951],[2101,2118],[2154,2167],[2214,2230]]
        video_path='/srv/glusterfs/gyglim/DATASETS/med_summaries_videos/HVC223640.mp4'
        self.run_test(true_shot_starts,video_path,'sewing video. (professional)')
    def testCutsNocuts5(self):
        true_shot_starts=[]
        video_path='/srv/glusterfs/gyglim/DATASETS/med_summaries_videos/HVC023529.mp4'
        self.run_test(true_shot_starts,video_path,'long video, mostly static')
if __name__=="__main__":
    unittest.main()


def getBlockHist(img,Params={'nbOfBlocksPerDim':2,'NbOfBins':16}):
        nbOfBlocksPerDim=Params['nbOfBlocksPerDim'] # number of blocks per axis
        NbOfBins=Params['NbOfBins'] # number of bins per histogram

        # initialize the histogram
        hist=np.zeros(nbOfBlocksPerDim*nbOfBlocksPerDim*NbOfBins*3)

        ''' Iterate over all nbOfBlocksPerDim x nbOfBlocksPerDim blocks '''
        for blockIdx1 in range(nbOfBlocksPerDim):
            # Compute the region y boundaries of this block
            y=[np.floor(img.shape[0]/nbOfBlocksPerDim)*(blockIdx1)+1, np.floor(img.shape[0]/nbOfBlocksPerDim)*(blockIdx1+1)]
            for blockIdx2 in range(0,nbOfBlocksPerDim):
                # Compute the x region boundaries of this block
                x=[np.floor(img.shape[1]/nbOfBlocksPerDim)*(blockIdx2)+1, np.floor(img.shape[1]/nbOfBlocksPerDim)*(blockIdx2+1)]

                # The position in the histogram
                startIdx=(blockIdx1*nbOfBlocksPerDim+blockIdx2)*NbOfBins*3

                for channel in range(0,3):
                    tmp=cv2.calcHist(img[y[0]:y[1],x[0]:x[1],channel],[0],None,[16],[0,255])

                    if tmp.sum()>0:
                        tmp=tmp/np.sum(tmp)
                    hist[startIdx+Params['NbOfBins']*channel:startIdx+Params['NbOfBins']*(channel+1)]=tmp.flatten()

        hist=hist/np.sum(hist)

        return hist

def getShots(video_path, threshold=0.08, verbose=False):

    cuts=[]
    #if verbose:



    # Get video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('File %s could not be opened' % video_path)


    # process it
    logger.info('Detect shots for video %s' % video_path)
    old_hist=None
    frame_idx=1
    long_step=2
    hist=list(False*np.zeros(long_step))
    change=[]
    while True:
        hasFrame, frame = cap.read()
        if hasFrame is False:
            break
        hist_idx=frame_idx % long_step
        hist[hist_idx]=getBlockHist(frame)
        if hist[hist_idx-1 % long_step] is not False and hist[hist_idx-(long_step-1) % long_step] is not False:
            diff=np.abs(hist[hist_idx-1 % long_step]-hist[hist_idx]).sum()
            diff_long=np.abs(hist[hist_idx-(long_step-1) % long_step]-hist[hist_idx]).sum()

            #change.append(diff*diff_long)
            change.append(diff)
            if diff > threshold and diff_long>threshold:
                cuts.append(frame_idx)

        frame_idx+=1
    cap.release()

    change_g=np.gradient(change)
    change_g=change_g/(0.5+5*np.percentile(np.abs(change),99))

    cuts=np.where(change_g>threshold)[0]
    cuts=cuts+long_step

    cuts_clean=[]
    for idx in range(0,len(cuts)-1):
        if cuts[idx]+1==cuts[idx+1]:
            continue
        else:
            cuts_clean.append(cuts[idx])
    if len(cuts)>0:
        cuts_clean.append(cuts[-1])
    if verbose:
        return cuts_clean,change_g
    else:
        return cuts_clean
