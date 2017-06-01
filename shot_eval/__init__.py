import re
import numpy as np
import glob
import os

from moviepy.editor import VideoFileClip
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

assert 'SHOT_DETECTION_DATASET' in os.environ, "Please download the ShotDetection Code & dataset from " \
                                    "http://imagelab.ing.unimore.it/files/ShotDetector.zip" \
                                    "and set SHOT_DETECTION_DATASET to its location "

DATASET_DIR = os.environ['SHOT_DETECTION_DATASET']

def eval(detection_fn, all=True):
    """Evaluates a detection function.

    Parameters
    ----------
    detection_fn
        A function that takes as input a video file path and returns a list of shots


    Returns
    -------
    f1_score : float
        the mean F1 score over the dataset
    """
    videos=sorted(glob.glob('%s/*.mp4' % DATASET_DIR))

    f1=[]
    for v in videos:
        videoid = os.path.splitext(os.path.split(v)[1])[0].split('_')[0]

        print "Run shot detection for %s" % videoid
        pred_shots=np.array(detection_fn(v))

        # Evaluate
        f1.append(get_f1(pred_shots,videoid))
        print f1[-1][0]

    f1=np.array(f1)
    print('Average frame error: %.2f' % np.mean(f1[:,1]))
    print('Median frame error: %.2f' % np.mean(f1[:,2]))
    print('Average precision: %.2f' % np.mean(f1[:, 3]))
    print('Average recall: %.2f' % np.mean(f1[:, 4]))
    print('Average F1: %.2f' % np.mean(f1[:, 0]))

    return np.mean(f1[:, 0])

def get_f1(detected_shots,video_id='25010', gt_shots=None, return_incorrect_shots=False):
    """Evaluation replicating evaluation of the C++ code of
    http://imagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19

    Parameters
    ----------
    detected_shots
        a list of detected shots
    video_id
        the id of the video

    Returns
    -------
    f1
    mean_frame_err
    median_frame_err

    """
    if gt_shots is None:
        gt_shots = np.loadtxt('%s/%s_gt.txt' % (DATASET_DIR,video_id)).astype('int')
    detected_shots=detected_shots.astype('int')
    gt_idx=0
    det_idx=0
    fp=0
    tp=0
    fn=0
    false_positives=[]
    false_negatives=[]
    frame_errors=[]
    detected_trans=[]
    for idx in range(1,len(detected_shots)):
        detected_trans.append([detected_shots[idx-1][1]+0.5,detected_shots[idx][0]-0.5])
    gt_trans=[]
    for idx in range(1,len(gt_shots)):
        gt_trans.append([gt_shots[idx-1][1]+0.5,gt_shots[idx][0]-0.5])

    while gt_idx<len(gt_trans) or det_idx<len(detected_trans):
        if gt_idx == len(gt_trans): # GT ended, but we have more transitions --> false positives
            false_positives.append(detected_trans[det_idx])
            fp+=1
            det_idx+=1
        elif det_idx == len(detected_trans): # Detected transition ended, but we have more GT transition --> false negatives
            false_negatives.append(gt_trans[gt_idx])
            fn+=1
            gt_idx+=1
        elif detected_trans[det_idx][1]< gt_trans[gt_idx][0]: #Detected transition ends before GT transition begins --> no overlap --> false positive
             false_positives.append(detected_trans[det_idx])
             fp+=1
             det_idx+=1
        elif detected_trans[det_idx][0] > gt_trans[gt_idx][1]:  # Detected transition begins after GT transition ends --> no overlap --> false negative
            false_negatives.append(gt_trans[gt_idx])
            fn+=1
            gt_idx+=1
        else: # GT and detected transition have an overlap >= 1, thus considered a hit aka true positive
            frame_error=np.abs(detected_trans[det_idx][0]-gt_trans[gt_idx][0]) + np.abs(detected_trans[det_idx][1]-gt_trans[gt_idx][1])
            frame_errors.append(frame_error/2.0)
            tp+=1
            gt_idx+=1
            det_idx+=1
    p = max(1e-8,tp) / max(1e-8,float(tp+fp))
    r = max(1e-8,tp) / max(1e-8,float(tp+fn))
    f1 = 2.0*r*p/max(1e-8,float(r+p))

    assert tp+fn==len(gt_trans)
    assert tp+fp==len(detected_trans)
    mean_frame_err=np.mean(frame_errors)
    median_frame_err=np.median(frame_errors)
    print('video_id: %s;\tTP: %d / %d - FP : %d - FN : %s	%.5fl;\t frame error: (%.2f; %.2f)' %
          (video_id,tp,tp+fn,fp,fn,f1,mean_frame_err,median_frame_err))

    if return_incorrect_shots:
        return false_positives, false_negatives
    else:
        return f1, mean_frame_err, median_frame_err, p, r


def visualize_errors(detected_shots, video_id='25010'):
    # Get the errors
    false_positives, false_negatives = get_f1(detected_shots, video_id, return_incorrect_shots=True)

    v = VideoFileClip('%s/%s.mp4' % (DATASET_DIR, video_id))

    for name,errors in zip(('FN','FP'),(false_negatives, false_positives)):
        for count, trans in enumerate(errors):
            plt.figure(figsize=(25,80))
            trans[0]=np.floor(trans[0])
            trans[-1] = np.ceil(trans[-1])
            for idx,frame_idx in enumerate(np.arange(trans[0],trans[1]+1)):
                plt.subplot(1,trans[1]-trans[0]+1,idx+1)
                plt.imshow(v.get_frame(frame_idx/v.fps))
                plt.title('%s; %.1f' % (name, frame_idx))
