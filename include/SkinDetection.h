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

    float mu_r;
    float mu_g;
    float mu_R;

    float sigma_r;
    float sigma_g;
    float sigma_R;



    void calculateRangeHist(Mat img, Mat mask, int &min, int &max, int minBinSize, string imTitle);
  public:
    SkinDetection();
    //SkinDetection(Mat image);
    bool init(Mat face, Mat face_mask);

    Mat getSkinMask(Mat image);
    int minBinSize;
    int minBinSizeLum;
    float scale;
};

#endif
