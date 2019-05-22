'''
Created on May 13, 2019

@author: maheriya
'''

import argparse
import numpy as np
import cv2 as cv
import sys

if sys.version_info[0] != 3:
    print("This module requires Python 3")
    sys.exit(1)


class Kalman3D:
    '''
    Kalman3D: Kalman filter to track a 3D point (x,y,z)
    All X Y Z should be in meters -- NOT in millimeters
    '''
    global debug, drag, grav, procNoise, measNoise
    ##-#######################################################################################    
    ## User-level Properties that can be changed for tuning
    ##-#######################################################################################    
    drag      = 1.0    # Drag felt by all axes (for example, air resistance)
    grav      = 9.801  # The constant acceleration on Y-axis
    procNoise = 0.1    # Process noise -- how good is our model?
    # 0.8: More uncertainty => more weight to prediction  (trust the model more)
    # 0.1: Less uncertainty => more weight to measurement (trust the measurement more)
    measNoise = 0.25    # Measurement noise: How good is the tracking?
    ##-#######################################################################################    



    ##-#######################################################################################    
    ## Properties that don't need to be changed for tuning
    global nstates, nmeasures, kf
    global SPX, SPY, SPZ, SVX, SVY, SVZ, SAC, MPX, MPY, MPZ
    global TDTX, TDTY, TDTZ, TDT2PX, TDT2PY, TDT2PZ, TDTVX, TDTVY, TDTVZ

    nstates   = 7 # px, py, pd, vx, vy, vd ay(6x6 matrix)
    nmeasures = 3 # All position data (p*), no velocities (v*)
    kf        = cv.KalmanFilter(nstates, nmeasures, 0)
    # State variable indices
    SPX = 0
    SPY = 1
    SPZ = 2
    SVX = 3
    SVY = 4
    SVZ = 5
    SAC = 6 # the constant acceleration; make sure to initialize it to a constant
    # Measurement variable indices
    MPX = 0
    MPY = 1
    MPZ = 2


    def __init__(self, drg=1.0, dbg=0):
        '''
        Params:
        drag: Drag coefficient. Use this to introduce drag. This is only an approximation
        '''
        global debug, drag
        drag           = drg
        debug          = dbg

        self.ticks     = 0     # keep track of time since last update (for dT)
        self.lastTicks = 0     # 
        if debug>=2: print("nstates", nstates)
        
        # A: Transition State Matrix -- the dynamics (constant acceleration model)
        # dT will be updated at each time stamp (could be fixed to; we are not using fixed dT)
        #    [PX   PY   PD   VX   VY   VD  AC  ]
        # px [ 1    0    0   dT    0    0 .5d2 ] (.5d2 --> 0.5 * dT**2)
        # py [ 0    1    0    0   dT    0 .5d2 ]
        # pd [ 0    0    1    0    0   dT .5d2 ]
        # vx [ 0    0    0   Drg   0    0   dT ]
        # vy [ 0    0    0    0   Drg   0   dT ]
        # vd [ 0    0    0    0    0   Drg  dT ]
        # ac [ 0    0    0    0    0    0   1  ]
        kf.transitionMatrix = np.eye(nstates, dtype=np.float32)
        kf.transitionMatrix[SVX, SVX] = drag
        kf.transitionMatrix[SVY, SVY] = drag
        kf.transitionMatrix[SVZ, SVZ] = drag
        if debug>=3: print("transitionMatrix: shape:{}\n{}".format(kf.transitionMatrix.shape, kf.transitionMatrix))

        # H: Measurement Matrix
        # [ 1 0 0 0 0 0 0] X
        # [ 0 1 0 0 0 0 0] Y
        # [ 0 0 1 0 0 0 0] D
        kf.measurementMatrix = np.eye(nmeasures, nstates, dtype=np.float32)
        if debug>=3: print("measurementMatrix: shape:{}\n{}".format(kf.measurementMatrix.shape, kf.measurementMatrix))
 
 
        # Q: Process Noise Covariance Matrix
        # [ Epx 0   0   0   0   0   0   ]
        # [ 0   Epy 0   0   0   0   0   ]
        # [ 0   0   Epd 0   0   0   0   ]
        # [ 0   0   0   Evx 0   0   0   ]
        # [ 0   0   0   0   Evy 0   0   ]
        # [ 0   0   0   0   0   Evd 0   ]
        # [ 0   0   0   0   0   0   Eac ]
        kf.processNoiseCov = np.eye(nstates, dtype=np.float32)*procNoise
        # Override errors for velocities (rely more on measurement for velocity rather than our model)
        kf.processNoiseCov[SVX, SVX] = 8.0;
        kf.processNoiseCov[SVY, SVY] = 8.0;
        kf.processNoiseCov[SVZ, SVZ] = 8.0;
        if debug>=3: print("processNoiseCov: shape:{}\n{}".format(kf.processNoiseCov.shape, kf.processNoiseCov))
        
        # R: Measurement Noise Covariance Matrix
        kf.measurementNoiseCov = np.eye(nmeasures, dtype=np.float32)*measNoise
        if debug>=3: print("measurementNoiseCov: shape:{}\n{}".format(kf.measurementNoiseCov.shape, kf.measurementNoiseCov))


    ## Public method 1/3
    def init(self, meas=np.float32([0,0,0])):
        '''
        Initialize the filter initial state
        Kalman filter actually doesn't have an init method. We just our hack our way through it.
        Call this init() before calling any other function of this class
        '''

        ## Initialize the filter (the Kalman Filter actually doesn't have any real method to initialize; we hack)
        state = np.zeros(kf.statePost.shape, np.float32)
        state[SPX] = meas[SPX];
        state[SPY] = meas[SPY];
        state[SPZ] = meas[SPZ];
        state[SVX] = 0.1;
        state[SVY] = 0.1;
        state[SVZ] = 0.1;
        state[SAC] = grav;
        kf.statePost = state;
        kf.statePre  = state;
        if debug>=2: print("statePost: shape:{}\n{}".format(kf.statePost.shape, kf.statePost))
        
        self.lastTicks = self.ticks;
        self.ticks = cv.getTickCount();
        return meas

    ## Public method 2/3
    ##-########################################################################################
    ## This is a convenience function that can be called every time a measurement
    ## is available or every time prediction is require based on past values.
    ##
    ##
    ## meas     : Measured 3D point
    ## onlyPred : Flag to ignore meas input (e.g., when input is not available)
    ## ret val  : Output predicted 3D point
    ##-########################################################################################
    def track(self, meas, dT=-1., onlyPred=False):
        '''
        User level function to do the tracking.
        meas: measurement data (ball position)
        Returns currently predicted (filtered) position of the ball
        '''
        if (onlyPred): ## only predict; ignore meas
            # This will be useful when there are no predictions available or
            # to predict future trajectory based on past measurements
            pred = self.Kpredict(dT)            # get predictions
            cpred = self.Kcorrect(pred, False)  # update with predicted values (restart means used pred value for correction 100% weight)
            if debug>=1:
                print("---------------------------------------------------")
                print("meas current               : None (only predicting)")
                print("pred predicted without meas: {}\n".format(cpred))
        else: # use meas to correct prediction
            pred = self.Kpredict(dT)           # get predictions
            cpred = self.Kcorrect(meas, False) # Kalman correct with measurement
            if debug>=1:
                print("---------------------------------------------------")
                print("meas current               : {}".format(meas))
                print("pred predicted             : {}\n".format(cpred))

        return cpred


    ## Public method 3/3
    ##-#######################################################################################
    ## Another convenience function that can be called to make predictions without
    ## measurements. Should only be used when measurement is not available.
    ## If measurement is available, use track() function which provides predicted value while
    ## updating the tracking state.
    ##
    ##
    ## pred: Output predicted 3D point
    ##-#######################################################################################
    def predict(self, dT=-1.):
        '''
        User level convenience function to do the prediction of trajectory.
        Returns predicted position
        '''
        return self.track(meas=None, dT=dT, onlyPred=True)


    ##-#######################################################################################
    ## Private methods
    def Kpredict(self, dT=-1.):
        '''
        Get predicted state. Each mat is a 3D point
        '''
        if (dT <= 0):
            self.lastTicks = self.ticks
            self.ticks = cv.getTickCount();
            dT = 1.0 * (self.ticks - self.lastTicks) / cv.getTickFrequency(); ## seconds

        if debug>=2: print("dT: {:1.4f}".format(dT))
        # Update the transition Matrix A with dT for this time stamp
        kf.transitionMatrix[SPX, SVX] = dT;
        kf.transitionMatrix[SPY, SVY] = dT;
        kf.transitionMatrix[SPZ, SVZ] = dT;
    
        #kf.transitionMatrix[SVX, SAC] = -dT;
        kf.transitionMatrix[SVY, SAC] = -dT;
        #kf.transitionMatrix[SVZ, SAC] = -dT;
        kf.transitionMatrix[SAC, SAC] = 1.;

        pred = kf.predict()
        return np.float32([pred[SPX], pred[SPY], pred[SPZ]]).squeeze()


    def Kcorrect(self, meas, restart=False):
        '''
        State correction using measurement matrix with 3D points
        '''
        if (restart): # Restart the filter
            # Initialization
            cv.setIdentity(kf.errorCovPre, 1.0);

            # Force the measurement to be used with 100% weight ignoring Hx
            kf.statePost[SPX] = meas[SPX];
            kf.statePost[SPY] = meas[SPY];
            kf.statePost[SPZ] = meas[SPZ];
            kf.statePost[SVX] = 3.;
            kf.statePost[SVY] = 3.;
            kf.statePost[SVZ] = 3.;
            kf.statePost[SAC] = grav;
        else:
            kf.correct(meas); # Kalman Correction
        return np.float32([kf.statePost[SPX], kf.statePost[SPY], kf.statePost[SPZ]]).squeeze()


    def getPostState(self):
        '''
        Get the state after correction
        '''
        return np.float32([kf.statePost[SPX], kf.statePost[SPY], kf.statePost[SPZ]]).squeeze()


#endclass
        
    