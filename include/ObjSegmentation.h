// FaceDetection.h
#ifndef OBJSEGMENTATION_H
#define OBJSEGMENTATION_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>



using namespace cv;
using namespace std;
using namespace pcl;



class ObjSegmentation
{
		private:
				Mat _image;
				void applyHystereses(cv::Mat image, float max_dist);
		public:
				ObjSegmentation(Mat image);
				void grabCut();
				void setSkinMask(std::vector<cv::Vec3b> ref_pixels, float t1, float t2);
				void applyMask(PointCloud<PointXYZRGB>::Ptr& cloud);

				Mat skin_mask;

};

#endif
