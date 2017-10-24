import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

def skeleton_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    #print c,r,w,h
	
    #cv2.imwrite('face.png',frame)
    #cv2.imwrite('sec.png',frame[r:r+h,c:c+w])
    #cv2.imwrite('sec1.png',frame[r:r+w,c:c+h])

    # Write track point for first frame
    pt=(frameCounter,c+w/2,r+h/2)
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1


    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    # initialize the tracker
	
    if file_name=='output_kalman.txt':
	state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position    
    	kf = cv2.KalmanFilter(4,2,0) # 4 state/hidden, 2 measurement, 0 control
	kf.transitionMatrix = np.array([[1., 0., .1, 0.],  # a rudimentary constant speed model:
                                    [0., 1., 0., .1],  # x_t+1 = x_t + v_t
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
	kf.measurementMatrix = 1. * np.eye(2, 4)      # you can tweak these to make the tracker
	kf.processNoiseCov = 1e-5 * np.eye(4, 4)      # respond faster to change and be less smooth
	kf.measurementNoiseCov = 1e-3 * np.eye(2, 2)
	kf.errorCovPost = 1e-1 * np.eye(4, 4)
	kf.statePost = state

    # or: particles = np.ones((n_particles, 2), int) * initial_pos
    elif file_name=='output_particle.txt':
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	hist_bp=cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    	n_particles = 200
	init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position	
	particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
	f0 = particleevaluator(hist_bp, init_pos) * np.ones(n_particles) # Evaluate appearance model
	weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)
	

    elif file_name=='output_of.txt':
	old_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
	lk_params = dict( winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
	mask = np.zeros_like(frame)

    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # perform the tracking
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()
	
        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        # write the result to the output file
	
	if file_name=='output_camshift.txt':
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
		ret, track_window = cv2.CamShift(dst, track_window, term_crit)
		(c,r,w,h)=track_window	
		pt=(frameCounter,c+w/2,r+h/2)
        	output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        
		#temp_file='/home/srajapurammo/assignment3/trash/'+str(frameCounter)+'.png'
		#cv2.imwrite(temp_file,frame[r:r+h,c:c+w])

	elif file_name=='output_kalman.txt':
		c,r,w,h=detect_one_face(frame)
		predict=kf.predict()
		if (c,r,w,h)==(0,0,0,0):
			pt=(frameCounter,int(predict[0][0]),int(predict[1][0]))
			#print 'b1'
		else:
			measurement=np.array([c+w/2,r+h/2],dtype='float64')
			posterior=kf.correct(measurement)
			pt=(frameCounter,c+w/2,r+h/2)
			#print 'b2'			

		#temp_file='/home/srajapurammo/assignment3/trash_kalman/'+str(frameCounter)+'.png'
		#fig=cv2.rectangle(frame,(pt[1]-5,pt[2]-5),(pt[1]+5,pt[2]+5),(255,0,0),2)
		#cv2.imwrite(temp_file,fig)
	
		output.write("%d,%d,%d\n" % pt)
	
	elif file_name=='output_particle.txt':
		stepsize=10
		np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")
		# Clip out-of-bounds particles
		im_w=len(frame[0])
		im_h=len(frame)
		particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)
		f = particleevaluator(hist_bp, particles.T) # Evaluate particles
		weights = np.float32(f.clip(1))             # Weight ~ histogram response
		weights /= np.sum(weights)                  # Normalize w
		pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

		if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    			particles = particles[resample(weights),:]  # Resample particles according to weights
		# resample() function is provided for you		
		#print weights
		#print pos
		
		pt=(frameCounter,pos[0],pos[1])
		output.write("%d,%d,%d\n" % pt)
		#temp_file='/home/srajapurammo/assignment3/trash_particle/'+str(frameCounter)+'.png'
		#fig=cv2.rectangle(frame,(pos[0]-10,pos[1]-10),(pos[0]+10,pos[1]+10),(255,0,0),2)
		#cv2.imwrite(temp_file,fig)	

	elif file_name=='output_of.txt':
		c,r,w,h=detect_one_face(frame)
		if (c,r,w,h)==(0,0,0,0):
			frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
			good_new = p1[st==1]
    			good_old = p0[st==1]			
			#print good_new
			#pt=(frameCounter,int(p1[0][0]),int(p1[0][1]))
			l=len(good_new)/2
			x=int(good_new[l][0])
			y=int(good_new[l][1])
			off_set=40
			pt=(frameCounter,x+off_set,y+off_set/7)		
			old_gray = frame_gray.copy()
			p0 = good_new.reshape(-1,1,2)				
				
		else:
			#frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)			
			pt=(frameCounter,c+w/2,r+h/2)
		
				
		#print pt
		#temp_file='/home/srajapurammo/assignment3/trash_of/'+str(frameCounter)+'.png'
		#fig=cv2.rectangle(frame,(pt[1]-5,pt[2]-5),(pt[1]+5,pt[2]+5),(255,0,0),2)
		#cv2.imwrite(temp_file,fig)
		output.write("%d,%d,%d\n" % pt)

	frameCounter = frameCounter + 1

    output.close()

if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        skeleton_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        skeleton_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        skeleton_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        skeleton_tracker(video, "output_of.txt")

'''
For Kalman Filter:

# --- init

state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state


# --- tracking

prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
'''

'''
For Particle Filter:

# --- init

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200

init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)


# --- tracking

# Particle motion model: uniform step (TODO: find a better motion model)
np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

# Clip out-of-bounds particles
particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

f = particleevaluator(hist_bp, particles.T) # Evaluate particles
weights = np.float32(f.clip(1))             # Weight ~ histogram response
weights /= np.sum(weights)                  # Normalize w
pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    particles = particles[resample(weights),:]  # Resample particles according to weights
# resample() function is provided for you
'''
