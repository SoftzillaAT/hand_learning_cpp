// SkinDetection.h
#ifndef SKINDETECTION_H
#define SKINDETECTION_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <math.h>
#include <stdint.h>
#include "PclManipulation.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#define PI 3.14159265

using namespace cv;
using namespace std;

class SkinDetection
{
  private:
    int _ymin;
    int _ymax;

    float _rLowBound;
    float _gLowBound;
    float _RLowBound;
    
    float _rUpBound;
    float _gUpBound;
    float _RUpBound;

    void calculateRangeHist(Mat img, Mat mask, int &min, int &max, string imTitle);
  public:
    SkinDetection();
    //SkinDetection(Mat image);
    bool init(Mat face, Mat face_mask);

    Mat getSkinMask(Mat image);
};

#endif
