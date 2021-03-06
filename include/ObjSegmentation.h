// ObjSegmentation.h
#ifndef OBJSEGMENTATION_H
#define OBJSEGMENTATION_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include "PclManipulation.h"
#include "Camera.h"
#include <math.h>

using namespace cv;
using namespace std;
using namespace pcl;



class ObjSegmentation
{
  private:
    Mat _image;
    Camera _cam;
    Point3f _obj_center;
    void applyHystereses(cv::Mat image, float max_dist);
  public:
    ObjSegmentation(Mat image, Camera cam);
    void grabCut(Mat mask);
    void setSkinMask(std::vector<cv::Vec3b> ref_pixels, uint8_t h_range, uint8_t s_range, uint8_t v_range, float max_hyst_dist );
    void applyMask(PointCloud<PointXYZRGB>::Ptr& cloud);
    void applyMask(PointCloud<PointXYZRGB>::Ptr& cloud, Mat mask);
    void clusterObject(PointCloud<PointXYZRGB>::Ptr& cloud, Point3f point);
    void clusterObjectWithGrabCut(PointCloud<PointXYZRGB>::Ptr& cloud);
    Point3f getObjCenter();
    Mat getMaskFromPcl(PointCloud<PointXYZRGB>::Ptr& cluser);
    cv::Point2d getNearestPoint(PointCloud<PointXYZRGB>::Ptr& cloud);
    cv::Point2d getNearestPoint(PointCloud<PointXYZRGB>::Ptr& cloud, Point3f refPoint);
    Mat skin_mask;
    Mat obj_mask;

    bool useGrabCut;

};

#endif
