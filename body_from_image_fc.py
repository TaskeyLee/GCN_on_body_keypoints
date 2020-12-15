# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
# import argparse

def body_points(image_path):
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
    
        # # Flags
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--image_path", default = "C:/Users/lab/Desktop/taskey/action_video_test/2.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        # # parser.add_argument("--video_path", default="examples/media/video.avi", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        # args = parser.parse_known_args()
    
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../../models/"
    
        # Add others in path?
        # for i in range(0, len(args[1])):
        #     curr_item = args[1][i]
        #     if i != len(args[1])-1: next_item = args[1][i+1]
        #     else: next_item = "1"
        #     if "--" in curr_item and "--" in next_item:
        #         key = curr_item.replace('-','')
        #         if key not in params:  params[key] = "1"
        #     elif "--" in curr_item and "--" not in next_item:
        #         key = curr_item.replace('-','')
        #         if key not in params: params[key] = next_item
    
        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()
    
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