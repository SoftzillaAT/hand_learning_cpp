#include "Camera.h"

Camera::Camera()
{
	init_camera = false;
  _rvec = cv::Mat::zeros(3, 1, CV_64F);
	_tvec = cv::Mat::zeros(3, 1, CV_64F);
}

Camera::Camera(Mat distCoeffs, Mat intrinsic)
{
	init_camera = false;
  _rvec = cv::Mat::zeros(3, 1, CV_64F);
	_tvec = cv::Mat::zeros(3, 1, CV_64F);
	_distCoeffs = distCoeffs;
	_intrinsic = intrinsic;
}

/*********************************************************************
 * Calculate the Transformation Matrix for projecting 3D Points to 2D Points
 ********************************************************************/
void Camera::calculateCameraSettings(PointCloud<PointXYZRGB>::Ptr& cloud)
{
				std::vector<cv::Point2d> imagePoints;
				std::vector<cv::Point3f> objectPoints;


				for (unsigned v = 0; v < cloud->height; v++)
				{
								for (unsigned u = 0; u < cloud->width; u++)
								{
												const pcl::PointXYZRGB &pt = (*cloud)(u,v);
												if (pt.z != pt.z || pt.x != pt.x || pt.y != pt.y)
												{
																continue;
												}
												imagePoints.push_back(cv::Point2d(u, v));
												objectPoints.push_back(cv::Point3f(pt.x, pt.y, pt.z));
								}
				}


				if(solvePnP(objectPoints, imagePoints, _intrinsic, _distCoeffs, _rvec, _tvec) )
				{
								cout << "Translation matrix calculated" << endl;
								init_camera = true;
				}

}


	std::vector<Point2f> Camera::projectPoints(std::vector<Point3f> points)
	{
		std::vector<Point2f> result;

		if (!init_camera)
		{
			cout << "Cameraparameters not set" << endl;
			return result;
		}
		cv::projectPoints(points, _rvec, _tvec, _intrinsic, _distCoeffs, result);
		return result;

	}
