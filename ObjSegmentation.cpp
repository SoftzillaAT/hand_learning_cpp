#include "ObjSegmentation.h"


ObjSegmentation::ObjSegmentation(Mat image, Camera cam)
{
  _image = image;
  _cam = cam;
  useGrabCut = false;
}

void ObjSegmentation::grabCut(Mat mask)
{

  cv::Rect rectangle;

  cv::Mat result = mask; // segmentation result (4 possible values)
  cv::Mat bgModel,fgModel; // the models (internally used)

  // GrabCut segmentation
  cv::grabCut(_image,    // input image
      mask,   // segmentation result
      rectangle,// rectangle containing foreground
      bgModel,fgModel, // models
      2,        // number of iterations
      cv::GC_INIT_WITH_MASK); // use rectangle

  cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
  cv::Mat foreground(_image.size(),CV_8UC3,cv::Scalar(255,255,255));
  _image.copyTo(foreground, result); // bg pixels not copied
  obj_mask = ~result;
  imshow("Segmentation", foreground);
}


void ObjSegmentation::setSkinMask(std::vector<cv::Vec3b> ref_pixels, uint8_t h_range, uint8_t s_range, uint8_t v_range, float max_hyst_dist)
{
  cv::Mat image_hsv;
  cv::cvtColor(_image, image_hsv, CV_BGR2HSV);
  skin_mask = Mat::zeros(_image.rows, _image.cols, CV_8UC1);

  for (int i = 0; i < ref_pixels.size(); i++)
  {
    cv::Vec3b ref = ref_pixels[i];
    uchar range = h_range;
    uchar range1 = s_range;
    uchar range2 = v_range;
    uchar l0 = (ref.val[0] - range < 0) ? 0 : ref.val[0] - range;
    uchar l1 = (ref.val[1] - range1 < 0) ? 0 : ref.val[1] - range1;
    uchar l2 = (ref.val[2] - range2 < 0) ? 0 : ref.val[2] - range2;

    uchar h0 = (ref.val[0] + range > 179) ? 179 : ref.val[0] + range;
    uchar h1 = (ref.val[1] + range1 > 255) ? 255 : ref.val[1] + range1;
    uchar h2 = (ref.val[2] + range2 > 255) ? 255 : ref.val[2] + range2;

    cv::Vec3b low(l0, l1, l2);
    cv::Vec3b high(h0, h1 , h2);

    //cv::Scalar low(0,10,50);
    //cv::Scalar high(170, 50, 100);

    //cout << "Low: " << low << " | High: " << high << endl;
    Mat mask;

    cv::inRange(image_hsv, low, high, mask);
    cv::bitwise_or(skin_mask, mask, skin_mask);

    // hue value is a circle so we need to check boundaries
    if (ref.val[0] - range < 0 )
    {
      l0 = 179 + (ref.val[0] - range);
      low[0] = l0;
      high[0] = 179;

      //cout << "u:Low: " << low << " | High: " << high << endl;
      cv::inRange(image_hsv, low, high, mask);
      cv::bitwise_or(skin_mask, mask, skin_mask);
    }
    if (ref.val[0] + range > 179 )
    {
      l0 = (ref.val[0] + range) % 180;
      high[0] = l0;
      low[0] = 0;

      //cout << "h:Low: " << low << " | High: " << high << endl;
      cv::inRange(image_hsv, low, high, mask);
      cv::bitwise_or(skin_mask, mask, skin_mask);
    }



  }

  cv::imshow("Hand Mask before hysterese", skin_mask);
  // get rid of single pixels
  //Mat gauss_mask;
  //GaussianBlur(skin_mask, gauss_mask, Size(5,5), 1);

  //skin_mask = gauss_mask;
  //cv::imshow("Gauß", gauss_mask);


  if (max_hyst_dist > 0)
  {
    applyHystereses(image_hsv, max_hyst_dist);
    cv::imshow("Hand Mask after hysterese", skin_mask);
  }
}


struct ColorDist {
  bool operator()(const Vec3b& a, const Vec3b &b) {
    return norm(a,b) < 65; // experimental value.
  }
};

// Lexicographic compare, same as for ordering words in a dictionnary:
// test first 'letter of the word' (x coordinate), if same, test 
// second 'letter' (y coordinate).
bool lexico_compare2d(const cv::Point2d& p1, const cv::Point2d& p2) {
  if(p1.x < p2.x) { return true; }
  if(p1.x > p2.x) { return false; }
  return (p1.y < p2.y);
}


bool points_are_equal2d(const cv::Point2d& p1, const cv::Point2d& p2) {
  return ((p1.x == p2.x) && (p1.y == p2.y));
}


void ObjSegmentation::applyHystereses(cv::Mat image_hsv, float max_dist)
{

  std::vector<cv::Point2d> queue; // queue of skin pixels
  Mat m_visited = Mat::zeros(_image.rows, _image.cols, CV_8UC1);
  
  for (int x = 0; x < image_hsv.cols; x++)
  {
    for (int y = 0; y < image_hsv.rows; y++)
    {
      cv::Point2d p(x,y);
      if (skin_mask.at<uchar>(p)>0)
      {
        queue.push_back(p);
      }
    }
  }

  while (queue.size() > 0)
  {
    cv::Point2d p = queue[0];
    queue.erase(queue.begin());

   if (m_visited.at<uint8_t>(p) > 0)
    continue;

    m_visited.at<uint8_t>(p) = 255;
    
    // iterate through neighbours
    for (int i = 0; i < 8; i++)
    {
      int x = p.x;
      int y = p.y;

      if (i == 0) { x = x - 1; y = y - 1; }
      if (i == 1) { y = y - 1; }
      if (i == 2) { x = x + 1; y = y - 1; }

      if (i == 3) { x = x - 1; }
      if (i == 4) { x = x + 1; }

      if (i == 5) { x = x - 1; y = y + 1;}
      if (i == 6) { y = y + 1; }
      if (i == 7) { x = x + 1; y = y + 1; }

      if ( x <= 0 || x >= skin_mask.cols || y <= 0 || y >= skin_mask.rows )
        continue;

      cv::Point2d p2(x, y);
     
      if (skin_mask.at<uchar>(p2) > 0)
        continue;

      cv::Vec3b color = image_hsv.at<cv::Vec3b>(p2);
      cv::Vec3b ref_color = image_hsv.at<cv::Vec3b>(p);
      
      float h1 = color[0];
      float s1 = color[1];
      float v1 = color[2];

      float h0 = ref_color[0];
      float s0 = ref_color[1];
      float v0 = ref_color[2];

      float dh = min(abs(h1-h0), 180-abs(h1-h0));
      float ds = abs(s1-s0);
      float dv = abs(v1-v0);
  
      // only try hs colorspace
      dv = 0;

      //float dist = norm(color, ref_color, cv::NORM_L2);
      float dist = sqrt(dh*dh+ds*ds+dv*dv);

      if (dist <= max_dist)
      {
        skin_mask.at<uchar>(p2) = 255;
        queue.push_back(p2);
      }
    }	
  }

}


void ObjSegmentation::clusterObject(PointCloud<PointXYZRGB>::Ptr& cloud, Point3f point)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  
  pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZRGB>(*cloud, *tmp_cloud);

  Mat kernel;
 
  // closing on skin mask
  Mat closing_mask;
  kernel = Mat::ones(19,19,CV_8UC1);
  cv::morphologyEx( skin_mask, closing_mask, 3, kernel );
  //cv::imshow("closing", closing_mask);

  Mat image_result;

  //  dilating the skin mask
  Mat output_mask = closing_mask;
  kernel = Mat::ones(13,13, CV_8UC1);
  cv::dilate(output_mask, output_mask, kernel);
  cv::imshow("dilating", output_mask);

  // apply mask for clustering
  applyMask(tmp_cloud, output_mask);

  cv::Point2d nearest_point = getNearestPoint(tmp_cloud, point);

  if (true)
  {
    Mat image_hsv;
    cv::cvtColor(_image, image_hsv, CV_BGR2HSV);
    obj_mask = PclManipulation::clusterCloud(tmp_cloud, nearest_point, image_hsv);
  }
  else
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster 
      (new pcl::PointCloud<pcl::PointXYZRGB>);
    PclManipulation::clusterCloud2(tmp_cloud, cluster);

    //Mat image;
    //_cam.convertImage(*cluster, image, cloud->width, cloud->height, 8); 
    //imshow("Clusterd cloud", image);

    obj_mask = getMaskFromPcl(cluster);
  }
  cv::imshow("cluster mask", obj_mask);
  Mat final_mask = closing_mask;
  cv::bitwise_or(final_mask, ~obj_mask, final_mask);
  final_mask = ~final_mask;

  // close holes in the mask with counours
  std::vector< std::vector<Point> > contours;
  std::vector<Vec4i> hierarchy;
  findContours( final_mask, contours, hierarchy, CV_RETR_TREE,
      CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  Scalar color = Scalar(255);
  Mat mask = Mat::zeros(_image.rows, _image.cols, CV_8UC1);
  drawContours( mask, contours, -1, color, CV_FILLED, 8, vector<Vec4i>(), 
      0, Point() );

  cv::imshow("Contour mask", mask);

  _image.copyTo(image_result, mask);
  cv::circle(image_result, nearest_point, 4, cv::Scalar(0,255,0), 2);
  cv::imshow("Obj Mask", image_result);


  if (!useGrabCut)
  {
    applyMask(cloud, ~mask);
  }
  else
  {
    // check if mask is valid
    int pixels = cv::countNonZero(mask);
    if (pixels < 100)
    {
      cout << "Too small mask of " << pixels << " pixels" << endl;
      applyMask(cloud, ~mask);
      return;
    }
    kernel = Mat::ones(21,21, CV_8UC1);
    cv::dilate(mask, output_mask, kernel);

    Mat grab_mask = skin_mask.clone();
    cv::bitwise_or(grab_mask, ~output_mask, grab_mask);

    grab_mask = ~grab_mask;

    cv::Mat foreground(_image.size(),CV_8UC3,cv::Scalar(255,255,255));
    _image.copyTo(foreground, grab_mask); // bg pixels not copied

    cv::imshow("Grabcut input", foreground);

    grab_mask.setTo(GC_PR_FGD, grab_mask == 255); 
    grabCut(grab_mask);
    applyMask(cloud, obj_mask);
  }

  Eigen::Matrix<float, 4, 1 > tmp;
  pcl::compute3DCentroid(*cloud , tmp);

  _obj_center.x = tmp[0];
  _obj_center.y = tmp[1];
  _obj_center.z = tmp[2];

}

Point3f ObjSegmentation::getObjCenter()
{
  return _obj_center;
}

Mat ObjSegmentation::getMaskFromPcl(PointCloud<PointXYZRGB>::Ptr& cluster)
{
  std::vector<cv::Point3f> points;

  BOOST_FOREACH (pcl::PointXYZRGB& pt, cluster->points)
  {
    points.push_back(cv::Point3f(pt.x, pt.y, pt.z));
  }

  std::vector<cv::Point2f> projectedPoints = _cam.projectPoints(points);


  std::vector<cv::Point2f> hull;
  cv::convexHull(projectedPoints, hull);


  std::vector<Point2i> hulli;
  for(int i = 0; i < hull.size(); i++)
    hulli.push_back( cv::Point2i( (int)round(hull[i].x), (int)round(hull[i].y) ) );


  vector<vector<Point2i> > contours;
  contours.push_back(hulli);
  Mat mask = Mat::zeros( _image.size(), CV_8UC1 );
  Scalar color = Scalar(255);
  drawContours( mask, contours, -1, color, CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
  return mask;
}


cv::Point2d ObjSegmentation::getNearestPoint(PointCloud<PointXYZRGB>::Ptr& cloud)
{
  return getNearestPoint(cloud, Point3f(0,0,0));
}

cv::Point2d ObjSegmentation::getNearestPoint(PointCloud<PointXYZRGB>::Ptr& cloud, Point3f refPoint)
{
  float minz = 10000;
  cv::Point result;
  for (unsigned v = 0; v < cloud->height; v++)
  {
    for (unsigned u = 0; u < cloud->width; u++)
    {
      pcl::PointXYZRGB &pt = (*cloud)(u,v);
      float x = pt.x - refPoint.x;
      float y = pt.y - refPoint.y;
      float z = pt.z - refPoint.z;
      float dist = sqrt(z * z + x * x + y * y);
      if (dist < minz)
      {
        minz = dist;
        result.x = u;
        result.y = v;
      }
    }
  }

  return result;
}


void ObjSegmentation::applyMask(PointCloud<PointXYZRGB>::Ptr& cloud)
{
  return applyMask(cloud, skin_mask);
}
void ObjSegmentation::applyMask(PointCloud<PointXYZRGB>::Ptr& cloud, Mat mask)
{	
  const float bad_point = std::numeric_limits<float>::quiet_NaN();

  for (unsigned v = 0; v < cloud->height; v++)
  {
    for (unsigned u = 0; u < cloud->width; u++)
    {

      pcl::PointXYZRGB &pt = (*cloud)(u,v);
      if( mask.at<bool>(v, u) > 0 )
      {
        pt.x = bad_point;
        pt.y = bad_point;
        pt.z = bad_point;
        pt.r = 0;
        pt.g = 0;
        pt.b = 0;
      }
    }
  }
}

