// FaceDetection.h
#ifndef FACEDETECTION_H
#define FACEDETECTION_H

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
#include "adaptiveskindetector.hpp"
#include "ColorHistogram.hpp"

#define PI 3.14159265

using namespace cv;
using namespace std;

class FaceDetection
{
  private:
    Mat _image_orig;
    Mat _face_image;
    Mat _face;
    Mat skin_mask;
    std::vector< cv::Vec3b > _skin_points;
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    String face_cascade_name;
    String eyes_cascade_name;

    void calcHistogram();
    std::vector<cv::Point3f> getHsvCylinder();
  public:
    FaceDetection(Mat image);
    void showResult();
    bool detectFace();
    std::vector< cv::Vec3b > getSkinPoints();

};

#endif
