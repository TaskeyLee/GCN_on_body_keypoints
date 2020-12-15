# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
# import argparse

def get_body_points(image_path):
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath('__file__'))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../../python/openpose/Release');
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../../x64/Release;' +  dir_path + '/../../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../../python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e
    
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../../models/"
    
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
    
        # Process Image
        datum = op.Datum()
        imageToProcess = cv2.imread(image_path)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        
        # Display Image
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        # cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
        # cv2.waitKey(0)
        return datum.poseKeypoints
    except Exception as e:
        print(e)
        sys.exit(-1)
    
# body_points("C:/Users/lab/Desktop/taskey/action_video_test/2.jpg")