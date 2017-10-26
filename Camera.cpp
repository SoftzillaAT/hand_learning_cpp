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





/*********************************************************************
 * This function converts a cloud image to an cv::Mat
 ********************************************************************/
void Camera::convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &_cloud, cv::Mat &_image)
{
  _image = cv::Mat_<cv::Vec3b>(_cloud.height, _cloud.width);

  for (unsigned v = 0; v < _cloud.height; v++)
  {
    for (unsigned u = 0; u < _cloud.width; u++)
    {
      cv::Vec3b &cv_pt = _image.at<cv::Vec3b> (v, u);
      const pcl::PointXYZRGB &pt = _cloud(u,v);

      cv_pt[2] = pt.r;
      cv_pt[1] = pt.g;
      cv_pt[0] = pt.b;
    }
  }
}

/*********************************************************************
 * This function converts a cloud image to an cv::Mat
 ********************************************************************/
void Camera::convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &_image, int width, int height, int radius = 1)
{
  _image = cv::Mat_<cv::Vec3b>(height, width);

  for (unsigned v = 0; v < height; v++)
  {
    for (unsigned u = 0; u < width; u++)
    {
      cv::Vec3b &cv_pt = _image.at<cv::Vec3b> (v, u);

      cv_pt[2] = 208;
      cv_pt[1] = 208;
      cv_pt[0] = 208;
    }
  }


  std::vector<cv::Point3f> objectPoints;
  std::vector<cv::Vec3b> rgbPoints;

  for (unsigned i = 0; i < cloud.points.size(); i++)
  {
    const pcl::PointXYZRGB &pt = cloud.points[i];
    if (pt.z != pt.z || pt.x != pt.x || pt.y != pt.y)
    {
      continue;
    }
    objectPoints.push_back(cv::Point3f(pt.x, pt.y, pt.z));
    rgbPoints.push_back(cv::Vec3b(pt.r, pt.g, pt.b));
  }

  std::vector<cv::Point2f> projectedPoints = projectPoints(objectPoints);

  for(unsigned i = 0; i < projectedPoints.size(); i++)
  {
    if(radius >= 1)
    {
      cv::circle(_image, projectedPoints[i], radius, cv::Scalar( rgbPoints[i][2], rgbPoints[i][1], rgbPoints[i][0] ), -1);
    }
    else
    {
      int x =  (int)round(projectedPoints[i].x);
      int y =  (int)round(projectedPoints[i].y);
      if(x < 0 || x >= cloud.width || y < 0 || y >= cloud.height)
        continue;
      cv::Vec3b &cv_pt = _image.at<cv::Vec3b> (y, x);
      cv_pt[2] = rgbPoints[i][0];
      cv_pt[1] = rgbPoints[i][1];
      cv_pt[0] = rgbPoints[i][2];
    }
  }
}


/*********************************************************************
 * This function converts a cloud image to an cv::Mat
 ********************************************************************/
void Camera::convertImage(const pcl::PointCloud<pcl::PointXYZRGBA> &_cloud, cv::Mat &_image)
{
  _image = cv::Mat_<cv::Vec3b>(_cloud.height, _cloud.width);

  for (unsigned v = 0; v < _cloud.height; v++)
  {
    for (unsigned u = 0; u < _cloud.width; u++)
    {
      cv::Vec3b &cv_pt = _image.at<cv::Vec3b> (v, u);
      const pcl::PointXYZRGBA &pt = _cloud(u,v);

      cv_pt[2] = pt.r;
      cv_pt[1] = pt.g;
      cv_pt[0] = pt.b;
    }
  }
}


/*********************************************************************
 * This function converts a cloud image to an cv::Mat
 ********************************************************************/
void Camera::convertImage(const pcl::PointCloud<pcl::PointXYZRGBNormal> &_cloud, cv::Mat &_image)
{
  _image = cv::Mat_<cv::Vec3b>(_cloud.height, _cloud.width);

  for (unsigned v = 0; v < _cloud.height; v++)
  {
    for (unsigned u = 0; u < _cloud.width; u++)
    {
      cv::Vec3b &cv_pt = _image.at<cv::Vec3b> (v, u);
      const pcl::PointXYZRGBNormal &pt = _cloud(u,v);

      cv_pt[2] = pt.r;
      cv_pt[1] = pt.g;
      cv_pt[0] = pt.b;
    }
  }
}

// draw 3D Point on img
void Camera::drawCircle(cv::Mat &img, cv::Point3f center, int radius, cv::Scalar color, int thickness)
{
				std::vector<cv::Point3f> tmpPoints;
				tmpPoints.push_back(center);
				std::vector<cv::Point2f> centerPoint = projectPoints(tmpPoints);
				cv::circle(img, centerPoint[0], radius, color, thickness);
}
