#include "PclManipulation.h"

pcl::PointCloud<PointXYZRGB>::Ptr PclManipulation::createCloud(std::vector<cv::Point3f> points)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  for (int i = 0; i < points.size(); i++)
  {
    PointXYZRGB p;
    p.x = points[i].x;
    p.y = points[i].y;
    p.z = points[i].z;
    p.r=255;
    p.g=1;
    p.b=1;
    cloud->points.push_back(p);
  }
  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

// Calc defined numbers of mean shift points over cloud
std::vector<cv::Point3f> PclManipulation::calcMeanShiftPoints(PointCloud<PointXYZRGB>::Ptr& cloud, int num_particles, float mean_shift_radius)
{
  std::vector<cv::Point3f> result;

  if (cloud->points.size() <= num_particles)
    num_particles = cloud->points.size();

  // initialize points for mean shift
  srand (time(NULL));
  if(cloud->points.size() > 0)
  {
    while(result.size() < num_particles)
    {
      pcl::PointXYZRGB& pt = cloud->points[rand() % cloud->points.size()];
      if(pt.r > 0)
      {
        result.push_back(cv::Point3f(pt.x, pt.y, pt.z));
      }
    }
  }

  for(int i = 0; i < result.size(); i++)
  {
    calcMeanShift(result[i],  cloud, mean_shift_radius);
  }

  return result;
}

// Calc mean shift
void PclManipulation::calcMeanShift(cv::Point3f &p, PointCloud<PointXYZRGB>::Ptr& cloud, float radius)
{
  cv::Point3f center (0,0,0);
  do
  {
    center = p;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr myRegion (new pcl::PointCloud<pcl::PointXYZRGB>);

    BOOST_FOREACH (pcl::PointXYZRGB& pt, cloud->points)
    {

      float dist = sqrt(pow(pt.x - p.x, 2) + pow(pt.y - p.y, 2) + pow(pt.z - p.z, 2));

      if(dist < radius)
        myRegion->push_back(pt);

    }

    myRegion->width = myRegion->points.size ();
    myRegion->height = 1;
    myRegion->is_dense = true;
    Eigen::Matrix<float, 4, 1 > tmp;
    pcl::compute3DCentroid(*myRegion , tmp);
    p.x = tmp[0];
    p.y = tmp[1];
    p.z = tmp[2];
  } while(p != center);

}

// get the center of the biggest cluster
cv::Point3f PclManipulation::getClusterPoint(std::vector<cv::Point3f> points, float cluster_dist)
{
  std::map<int, std::vector<cv::Point3f> > clusters;
  for(int i = 0; i < points.size(); i++)
  {
    for(int j = 0; j < points.size(); j++)
    {
      if(clusters[j].size() == 0)
      {
        clusters[j].push_back(points[i]);
        break;
      }
      else
      {
        std::vector<cv::Point3f> p = clusters[j];
        cv::Point3f p1 = points[i];
        cv::Point3f p2 = p[0];
        float dist = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));

        if(dist < cluster_dist)
        {
          clusters[j].push_back(points[i]);
          break;
        }
      }

    }
  }

  // get cluster with max points
  int max_size = 0;
  cv::Point3f result;
  for(int i=0; i < clusters.size(); i++)
  {
    int r = rand() % 255;
    int g = rand() % 255;
    int b = rand() % 255;

    for(int j = 0; j < clusters[i].size(); j++)
    {
      //drawCircle(image, clusters[i][j], 20, cv::Scalar( b, g, r ), 4);
    }

    if(clusters[i].size() > max_size)
    {

      max_size = clusters[i].size();
      std::vector<cv::Point3f> p = clusters[i];
      result = p[p.size() - 1];
    }
  }
  //cv::namedWindow( "calibration", CV_WINDOW_AUTOSIZE );
  //cv::imshow("calibration",image);

  return result;
}


Mat PclManipulation::clusterCloud(pcl::PointCloud<PointXYZRGB>::Ptr& cloud, cv::Point2d p, Mat image_hsv)
{
  std::vector<cv::Point2d> queue; // queue of skin pixels
  Mat m_visited = Mat::zeros(cloud->height, cloud->width, CV_8UC1);
  Mat obj_mask = Mat::zeros(cloud->height, cloud->width, CV_8UC1);
  
  queue.push_back(p);

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

      if ( x <= 0 || x >= obj_mask.cols || y <= 0 || y >= obj_mask.rows )
        continue;

      cv::Point2d p2(x, y);

      if (obj_mask.at<uchar>(p2) > 0)
        continue;

      pcl::PointXYZRGB &pt_nb = (*cloud)(p2.x, p2.y);
      pcl::PointXYZRGB &pt_ref = (*cloud)(p.x, p.y);

      float dist = norm(cv::Vec3f(pt_nb.x, pt_nb.y, pt_nb.z), cv::Vec3f(pt_ref.x, pt_ref.y, pt_ref.z), cv::NORM_L2);

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

      //float dist = norm(color, ref_color, cv::NORM_L2);
      float color_dist = sqrt(dh*dh+ds*ds+dv*dv);
      bool addPoint = false;
      
      //cout << "COLOR DISTANCE: " << color_dist << endl;
      if (dist <= 0.03 || ( dist <= 0.1 && color_dist <= 8) )
      {
        addPoint = true;
      }

      if (addPoint)
      {
        obj_mask.at<uchar>(p2) = 255;
        queue.push_back(p2);
      }
    }	
  }

  return obj_mask;
}

// Lexicographic compare, same as for ordering words in a dictionnary:
// test first 'letter of the word' (x coordinate), if same, test 
// second 'letter' (y coordinate).
bool PclManipulation::lexico_compare2d(const cv::Point2d& p1, const cv::Point2d& p2) {
  if(p1.x < p2.x) { return true; }
  if(p1.x > p2.x) { return false; }
  return (p1.y < p2.y);
}


bool PclManipulation::points_are_equal2d(const cv::Point2d& p1, const cv::Point2d& p2) {
  return ((p1.x == p2.x) && (p1.y == p2.y));
}


std::vector<int> PclManipulation::getNearestNeigboursInRadius(pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree, float radius, cv::Point3f p)
{
  pcl::PointXYZRGB searchPoint;

  if (p.x != p.x || p.y != p.y || p.z != p.z)
  {
    cout << "getNearestNeigboursInRadius: ERROR bad point";
    std::vector<int> result;
    return result;
  }

  searchPoint.x = p.x;
  searchPoint.y = p.y;
  searchPoint.z = p.z;

  // K nearest neighbor search

  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;


  kdtree.radiusSearch (searchPoint, radius, pointIdxNKNSearch, pointNKNSquaredDistance);



  return pointIdxNKNSearch;
}



void PclManipulation::clusterCloud2(PointCloud<PointXYZRGB>::Ptr& cloud, PointCloud<PointXYZRGB>::Ptr& cloud_result)
{

  // Create 2nd Pointcloud for manipulation
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);

  // Remove NaN points from the cloud
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, indices);

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZRGB> vg;

  vg.setInputCloud (cloud_filtered);
  vg.setLeafSize (0.03f, 0.03f, 0.03f);
  vg.filter (*cloud_filtered);
  //std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*


  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance (0.04); // 5cm
  ec.setMinClusterSize (5);
  ec.setMaxClusterSize (2500000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_all (new pcl::PointCloud<pcl::PointXYZRGB>);
  int j = 0;

  vector < pcl::PointCloud<pcl::PointXYZRGB>, Eigen::aligned_allocator <pcl::PointCloud <pcl::PointXYZRGB> > > cloud_clusters;

  /* initialize random seed: */
  srand (time(NULL));

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    int r = rand() % 255;
    int g = rand() % 255;
    int b = rand() % 255;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr mycluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    {
      mycluster->points.push_back(cloud_filtered->points[*pit]);
      cloud_filtered->points[*pit].r=r;
      cloud_filtered->points[*pit].g=g;
      cloud_filtered->points[*pit].b=b;
      cloud_cluster_all->points.push_back (cloud_filtered->points[*pit]);
    }
    mycluster->width = mycluster->points.size();
    mycluster->height = 1;
    mycluster->is_dense = true;
    cloud_clusters.push_back(*mycluster);

    j++;
  }

  cloud_cluster_all->width = cloud_cluster_all->points.size ();
  cloud_cluster_all->height = 1;
  cloud_cluster_all->is_dense = true;

  Eigen::Matrix<float, 4, 1 > center;
  double min_dist = 99999;
  int result_cluster_index = 0;

  for(int i = 0; i < cloud_clusters.size(); i++)
  {
    pcl::compute3DCentroid	(cloud_clusters[i] , center);
    /*
       unsigned int pcl::compute3DCentroid	(	const pcl::PointCloud< PointT > & 	cloud,
       const pcl::PointIndices & 	indices,
       Eigen::Matrix< Scalar, 4, 1 > & 	centroid 
       )	
       */
    double dist = sqrt(pow(center[0],2) + pow(center[1],2) + pow(center[2],2));
    if (dist < min_dist)
    {
      min_dist = dist;
      result_cluster_index = i;
    }	
  }




  if (cloud_cluster_all->points.size () > 0)
  {

    for(int i = 0; i <  cloud_clusters[result_cluster_index].points.size(); i++)
    {
      cloud_result->points.push_back(cloud_clusters[result_cluster_index].points[i]);
    }

    cloud_result->width = cloud_result->points.size ();
    //cout << "CLOUD CLUSTERED " << cloud_result->points.size() << endl;
    cloud_result->height = 1;
    cloud_result->is_dense = true;
  }
}

