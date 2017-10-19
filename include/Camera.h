#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>
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
};


#endif //CAMERA_H
