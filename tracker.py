'''
3D Tracker: Takes an array of X Y Z position values as input and 
outputs an array of predicted trajectory.
All X Y Z should be in meters -- NOT in millimeters
'''
import argparse
import numpy as np
import cv2 as cv
import sys
from filter.kalman import Kalman3D
if sys.version_info[0] != 3:
    print("This script requires Python 3")
    sys.exit(1)

DEBUG = 0    


if __name__ == '__main__':
    fmt = lambda x: "%8.3f" % x
    np.set_printoptions(formatter={'float_kind':fmt})

    parser = argparse.ArgumentParser(description='''
Find predicted trajectory based on past positions. For example:
        python3 tracker --n 10 --pos 1265 479 5032

This script will print the output trajectory.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('positions', metavar='X Y Z', nargs='+', type=float,
                        help='Coordinates of ball positions as triplets of X Y Z values. Make sure to provide enough past measurements to ensure correct tracking')
    parser.add_argument('--n', dest='npred', type=int, default=1,
                        help='Number of predicted position values in the output trajectory')

    args = parser.parse_args()

    try:
        res = len(args.positions)
    except:
        print("Please provide input position points as triplets of X Y Z values")
        parser.print_help()
        sys.exit()
    else:
        if (res%3 != 0 ):
            print("Please provide input position points as triplets of X Y Z values")
            parser.print_help()
            sys.exit()
    ## Convert to array of measurements in meters
    positions = np.float32(args.positions)/1000.
    numpos = len(args.positions)//3
    print("{} input positions given.".format(numpos))
    positions = positions.reshape(numpos, 3)
    if DEBUG:
        print("positions shape", positions.shape)
        print(positions)
    ##-#######################################################################################
    ## Instantiate Kalman3D tracker
    ##-#######################################################################################
    KF = Kalman3D(drg=1.0, dbg=0)
    KF.init(positions[0])

    ##-#######################################################################################
    ## Tracking
    ## Since we are doing all operations in zero time, specify dT manually (e.g., 0.033 sec)
    for i in range(positions.shape[0]-1):
        pred = KF.track(positions[i+1], 0.033)*1000
        print("  tracked position : {}".format(pred))

    
    ##-#######################################################################################
    ## Trajectory prediction
    ## Since we are doing all operations in zero time, specify dT manually (e.g., 0.033 sec)
    for i in range(args.npred):
        pred = KF.predict(0.033)*1000
        print("predicted position : {}".format(pred))
        













#EOF
