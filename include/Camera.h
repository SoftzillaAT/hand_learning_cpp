#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;
using namespace pcl;

class Camera
{
  private:
    cv::Mat _rvec;
    cv::Mat _tvec;
    cv::Mat_<double> _distCoeffs;
    cv::Mat_<double> _intrinsic;
  public:
    Camera();
    Camera(Mat distCoeffs, Mat intrinsic);
    bool init_camera;
    void calculateCameraSettings(PointCloud<PointXYZRGB>::Ptr& cloud);
    std::vector<Point2f> projectPoints(std::vector<Point3f> points);
    void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &image);
    void convertImage(const pcl::PointCloud<pcl::PointXYZRGBA> &_cloud, cv::Mat &_image);
    void convertImage(const pcl::PointCloud<pcl::PointXYZRGBNormal> &_cloud, cv::Mat &_image);
    void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &_cloud, cv::Mat &_image, int width, int height, int radius);
    void drawCircle(cv::Mat &img, cv::Point3f center, int radius, cv::Scalar color, int thickness);
};


#endif //CAMERA_H
