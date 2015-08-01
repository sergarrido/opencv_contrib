/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/


#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;


/**
 */
static void help() {
    cout << "Basic marker detection" << endl;
    cout << "Parameters: " << endl;
    cout << "-d <dictionary> # 0: ARUCO, ..." << endl;
    cout << "[-v <videoFile>] # Input from video file, if ommited, input comes from camera"
                 << endl;
    cout << "[-ci <int>] # Camera id if input doesnt come from video (-v). Default is 0"
                 << endl;
    cout << "[-c <cameraParams>] # Camera intrinsic parameters. Needed for camera pose"
              << endl;
    cout << "[-l <markerLength>] # Marker side lenght (in meters). Needed for correct" <<
                 "scale in camera pose, default 0.1" << endl;
    cout << "[-dp <detectorParams>] # File of marker detector parameters" << endl;
    cout << "[-r] # show rejected candidates too" << endl;
}



/**
 */
static bool isParam(string param, int argc, char **argv ) {
    for (int i=0; i<argc; i++)
        if (string(argv[i]) == param )
            return true;
    return false;

}


/**
 */
static string getParam(string param, int argc, char **argv, string defvalue = "") {
    int idx=-1;
    for (int i=0; i<argc && idx==-1; i++)
        if (string(argv[i]) == param)
            idx = i;
    if (idx == -1 || (idx + 1) >= argc)
        return defvalue;
    else
        return argv[idx+1];
}



/**
 */
static void readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
}



/**
 */
static void readDetectorParameters(string filename, aruco::DetectorParameters &params) {
    FileStorage fs(filename, FileStorage::READ);
    fs["adaptiveThreshWinSize"] >> params.adaptiveThreshWinSize;
    fs["adaptiveThreshConstant"] >> params.adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params.minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params.maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params.polygonalApproxAccuracyRate;
    fs["minCornerDistance"] >> params.minCornerDistance;
    fs["minDistanceToBorder"] >> params.minDistanceToBorder;
    fs["minMarkerDistance"] >> params.minMarkerDistance;
    fs["doCornerRefinement"] >> params.doCornerRefinement;
    fs["cornerRefinementWinSize"] >> params.cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params.cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params.cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params.markerBorderBits;
    fs["perspectiveRemoveDistortion"] >> params.perspectiveRemoveDistortion;
    fs["perspectiveRemovePixelPerCell"] >> params.perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params.perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params.maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params.minOtsuStdDev;
    fs["errorCorrectionRate"] >> params.errorCorrectionRate;
}



/**
 */
int main(int argc, char *argv[]) {

    if (!isParam("-d", argc, argv)) {
        help();
        return 0;
    }

    int dictionaryId = atoi( getParam("-d", argc, argv).c_str() );
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(
                                   aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    bool showRejected = false;
    if (isParam("-r", argc, argv))
      showRejected = true;

    bool estimatePose = false;
    Mat camMatrix, distCoeffs;
    if (isParam("-c", argc, argv)) {
      readCameraParameters(getParam("-c", argc, argv), camMatrix, distCoeffs);
      estimatePose = true;
    }
    float markerLength = (float)atof( getParam("-l", argc, argv, "0.1").c_str() );

    aruco::DetectorParameters detectorParams;
    if (isParam("-dp", argc, argv)) {
      readDetectorParameters(getParam("-dp", argc, argv), detectorParams);
    }

    VideoCapture inputVideo;
    int waitTime;
    if (isParam("-v", argc, argv)) {
        inputVideo.open(getParam("-v", argc, argv));
        waitTime = 0;
    }
    else {
        int camId = 0;
        if (isParam("-ci", argc, argv))
            camId = atoi( getParam("-ci", argc, argv).c_str() );
        inputVideo.open(camId);
        waitTime = 10;
    }

    double totalTime = 0;
    int totalIterations = 0;

    while (inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        double tick = (double)getTickCount();

        vector<int> ids;
        vector<vector<Point2f> > corners, rejected;
        vector<Mat> rvecs, tvecs;

        // detect markers and estimate pose
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
        if (estimatePose && ids.size() > 0)
            aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs,
                                                 rvecs, tvecs);

        double currentTime = ((double)getTickCount()-tick)/getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations%30 == 0) {
            cout << "Detection Time = " << currentTime*1000 << " ms " <<
                         "(Mean = " << 1000*totalTime/double(totalIterations) << " ms)" << endl;
        }

        // draw results
        image.copyTo(imageCopy);
        if (ids.size() > 0) {
            aruco::drawDetectedMarkers(imageCopy, imageCopy, corners, ids);

            if (estimatePose) {
                for (unsigned int i = 0; i < ids.size(); i++)
                        aruco::drawAxis(imageCopy, imageCopy, camMatrix, distCoeffs, rvecs[i],
                                            tvecs[i], markerLength*0.5f);
            }
        }

        if (showRejected && rejected.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, imageCopy, rejected,
                                               noArray(), Scalar(100, 0, 255));

        imshow("out", imageCopy);
        char key = (char) waitKey(waitTime);
        if (key == 27)
            break;
    }

    return 0;
}
