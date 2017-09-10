// Modeling Tool for in-hand-objects
// written by Dominik Streicher
// based on Hannes Prankl camera_tracking_and_mapping

// TODO Optimizitation: Kdtree calculation more than once, remove NAN more than once
// TODO to find the object use the property of the edge of the clustered object

#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>

#include <iostream>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/centroid.h>


#include <cv.h>
#include <highgui.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/min_cut_segmentation.h>

#include <utility>                   // for std::pair
#include <algorithm>                 // for std::for_each
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/astar_search.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <pcl/kdtree/kdtree_flann.h>

/* Includes of Hannes */
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <queue>
#include <unistd.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/common/time.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/io/ply_io.h>
#include <pcl/conversions.h>

#include "pcl/common/transforms.h"

#include "v4r/keypoints/impl/PoseIO.hpp"
#include "v4r/keypoints/impl/invPose.hpp"
#include "v4r/camera_tracking_and_mapping/TSFVisualSLAM.h"
#include "v4r/camera_tracking_and_mapping/TSFData.h"
#include "v4r/reconstruction/impl/projectPointToImage.hpp"
#include "v4r/features/FeatureDetector_KD_FAST_IMGD.h"
#include "v4r/camera_tracking_and_mapping/TSFGlobalCloudFilteringSimple.h"
#include "v4r/io/filesystem.h"

using namespace std;
using namespace pcl;

/* */

typedef boost::property<boost::edge_weight_t, int> EdgeWeightProperty;
typedef boost::adjacency_list<boost::listS, boost::vecS,boost::undirectedS,boost::no_property,EdgeWeightProperty> UndirectedGraph;
typedef boost::graph_traits<UndirectedGraph>::edge_iterator edge_iterator;

/* */

/* ######################### Macros ############################### */
#if DEBUG_LEVEL == 2  
	#define DEBUG(l,x) if(l<=2) x
#endif
#if DEBUG_LEVEL == 1  
	#define DEBUG(l,x) if(l<=1) x
#endif
#if DEBUG_LEVEL == 0
	#define DEBUG(l,x)
#endif 


/* ######################### Methods ############################## */
void printUsage(const char*);
void grabberCallback(const PointCloud<PointXYZRGBA>::ConstPtr&);
void calculateCameraSettings(PointCloud<PointXYZRGB>::Ptr& cloud);
void removeBackground(PointCloud<PointXYZRGB>::Ptr&, PointCloud<PointXYZRGB>::Ptr&);
std::vector<cv::Point3f> calcMeanShiftPoints(PointCloud<PointXYZRGB>::Ptr&);
void postProcessing(PointCloud<PointXYZRGB>::Ptr&);
void calcMeanShift(cv::Point3f &p, PointCloud<PointXYZRGB>::Ptr&, float radius);
cv::Point3f getCenterPoint(std::vector<cv::Point3f> points);
cv::Point3f getClusterPoint(std::vector<cv::Point3f> points);
void removeHand(PointCloud<PointXYZRGB>::Ptr&, cv::Point3f roi_center, PointCloud<PointXYZRGB>::Ptr&); 
std::vector<int> getKNearestNeigbours(pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree, int K, cv::Point3f p); // get k nearest neighbours and returns their index
std::vector<int> getNearestNeigboursInRadius(pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree, float radius, cv::Point3f p);
std::vector<int> getShortestPath(PointCloud<PointXYZRGB>::Ptr&, cv::Point3f start, cv::Point3f end, float maxDist); // maxDist=maximum distance between 2 neighbours

void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &image);
void convertImage(const pcl::PointCloud<pcl::PointXYZRGBA> &_cloud, cv::Mat &_image);
void convertImage(const pcl::PointCloud<pcl::PointXYZRGBNormal> &_cloud, cv::Mat &_image);
void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &_cloud, cv::Mat &_image, int width, int height, int radius);


void drawCoordinateSystem(cv::Mat &im, const Eigen::Matrix4f &pose, const cv::Mat_<double> &intrinsic, const cv::Mat_<double> &dist_coeffs, double size, int thickness);
void drawConfidenceBar(cv::Mat &im, const double &conf, int x_start=50, int x_end=200, int y=30);
void drawCircle(cv::Mat &img, cv::Point3f center, int radius, cv::Scalar color, int thickness);

bool lexico_compare2f(const cv::Point2f& p1, const cv::Point2f& p2);
bool points_are_equal2f(const cv::Point2f& p1, const cv::Point2f& p2);
bool lexico_compare3f(const cv::Point3f& p1, const cv::Point3f& p2);
bool points_are_equal3f(const cv::Point3f& p1, const cv::Point3f& p2);

void initTracker();
void stopTracker();
void trackImage(const PointCloud<PointXYZRGB>::ConstPtr&);


/* ######################### Constants ############################# */
const float bad_point = std::numeric_limits<float>::quiet_NaN();
const int max_loose_pose = 10; 						// Maximum number of lost poses until the grabber ends


/* ######################### Variables ############################# */
PointCloud<PointXYZRGB>::Ptr mycloud (new PointCloud<PointXYZRGB>); 	// A cloud that will store color info.
Grabber* openniGrabber;                                               	// OpenNI grabber that takes data from the device.
unsigned int filesSaved = 0;                                          	// For the numbering of the clouds saved to disk.
bool stopCamera(false);							// Stop the camera callback
enum Mode { capture, calibration, tracking, stop };					// Current mode
Mode mode = capture;							// Start with capturing images

v4r::TSFVisualSLAM tsf;
std::vector< std::pair<Eigen::Matrix4f, int> > all_poses; 		//<pose, kf_index>
pcl::PointCloud<pcl::PointXYZRGBNormal> filt_cloud;
Eigen::Matrix4f filt_pose;
Eigen::Matrix4f inv_pose;

double conf_ransac_iter;
double conf_tracked_points;
uint64_t ts_last;
int i;									// index of the current tracking
bool have_pose(true);
int lost_pose_counter;
double mytime;
double mean_time;
int cnt_time;
uint64_t timestamp;

cv::Mat_<cv::Vec3b> image;
cv::Mat_<cv::Vec3b> im_draw;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

std::string cam_file, filenames;
std::string file_mesh = "mesh.ply";
std::string file_cloud = "cloud.pcd";

cv::Mat_<double> distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
cv::Mat_<double> intrinsic = cv::Mat_<double>::eye(3,3);
cv::Mat_<double> dist_coeffs_opti = cv::Mat::zeros(4, 1, CV_64F);
cv::Mat_<double> intrinsic_opti = cv::Mat_<double>::eye(3,3);

Eigen::Matrix4f pose;
float voxel_size = 0.0005;//0.0005;
double thr_weight = 2;      //e.g. 10    // surfel threshold for the final model
double thr_delta_angle = 80; // e.g. 80
int poisson_depth = 6;
int display = true;

int max_camera_distance = 150; // in [cm]
int num_particles = 20;
int num_calibration_frames = 10;
int cnt_calibration_frames;

cv::Point track_win[2];

int mean_shift_radius = 15; // in cm
cv::Point3f roi_center;
std::vector<cv::Point3f> roiPoints;

// camera parameter
bool init_camera = false;
cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);


// min cut parameters
int mc_sigma=0.25*100;
int mc_radius=0.3*100;
int mc_numberOfNeighbours=7;
int mc_sourceWeight =0.6*100;

/*********************************************************************
 * Main entrypoint of the program
 ********************************************************************/
int main(int argc, char** argv)
{
	DEBUG(1, cout << "DEBUG_LEVEL 1 enabled" << endl);
  	DEBUG(2, cout << "DEBUG_LEVEL 2 enabled" << endl);
	if (console::find_argument(argc, argv, "-h") >= 0)
	{
		printUsage(argv[0]);
		return -1;
	}


	openniGrabber = new OpenNIGrabber();
	if (openniGrabber == 0)
		return -1;
	boost::function<void (const PointCloud<PointXYZRGBA>::ConstPtr&)> f =
		boost::bind(&grabberCallback, _1);
	openniGrabber->registerCallback(f);
	
	// gui for the visualization
	cv::namedWindow( "image", CV_WINDOW_AUTOSIZE );
	cv::namedWindow( "orig", CV_WINDOW_AUTOSIZE );
	
	
	// camera settings
	intrinsic << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics
	
	// initialize tracker
	initTracker();
 	
	// start callback for the camera images
	openniGrabber->start();
	
	// trackbars
	cv::namedWindow("trackbars",CV_WINDOW_KEEPRATIO);
	cv::createTrackbar("Max camera distance [cm]","trackbars", &max_camera_distance, 400);
	cv::createTrackbar("Mean_shift radius [cm]","trackbars", &mean_shift_radius, 100);
	cv::createTrackbar("Number of patricles","trackbars", &num_particles, 200);
	cv::createTrackbar("Number of calibraion frames","trackbars", &num_calibration_frames, 100);
	
	cv::createTrackbar("MIN CUT: sigma","trackbars", &mc_sigma, 1000);
	cv::createTrackbar("MIN CUT: radius","trackbars", &mc_radius, 500);
	cv::createTrackbar("MIN CUT: number of neighbours","trackbars", &mc_numberOfNeighbours, 500);
	cv::createTrackbar("MIN CUT: source weight","trackbars", &mc_sourceWeight, 1000);
	

	while(!stopCamera)
	{
		boost::this_thread::sleep(boost::posix_time::seconds(1));
		DEBUG(2, cout << "Check stop camera: " << stopCamera << endl);
	}
	
	DEBUG(1, cout << "Stoping OpenNi" << endl);
	// stop the camera 
	openniGrabber->stop();
	
	// stop the tracker if something caught tracked
	if (i > 0)
		stopTracker();
	
	DEBUG(1, cout<<"Finished!"<<endl);


	return 0;
}


/*********************************************************************
 * Print usage of the program
 ********************************************************************/
void printUsage(const char* programName)
{
	cout << "Usage: " << programName << " [options]"
		 << endl
		 << endl
		 << "Options:\n"
		 << endl
		 << "\t<none>     start capturing from an OpenNI device.\n"
		 << "\t-v FILE    visualize the given .pcd file.\n"
		 << "\t-h         shows this help.\n";
}



/*********************************************************************
 * This function is called every time the device has new data.
 ********************************************************************/
void grabberCallback(const PointCloud<PointXYZRGBA>::ConstPtr& cloud)
{
	clock_t gstart, gend, start, end;
	gstart = clock();
	start = clock();
	DEBUG(2, cout << "Callback..." << endl);
	// copy to the pcl for write-access
	pcl::copyPointCloud<pcl::PointXYZRGBA, pcl::PointXYZRGB>(*cloud, *mycloud);
	
	if(!init_camera)
	{
		 calculateCameraSettings(mycloud);
	}
	
	//convertImage(*mycloud, image);
    	convertImage(*mycloud, image, mycloud->width, mycloud->height, 0);
    	
    	image.copyTo(im_draw);
	cv::imshow("orig",im_draw);
	end = clock();
	cout << "Time required for init: "<< (double)(end-start)/CLOCKS_PER_SEC << " seconds." << "\n\n";
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

	start = clock();
	
	removeBackground(mycloud, result_cluster);
	end = clock();
	cout << "Time required for remove background: "<< (double)(end-start)/CLOCKS_PER_SEC << " seconds." << "\n\n";
	
	
	
	int key= cv::waitKey(10);
	if (mode == capture)
	{
		DEBUG(2, cout << "Mode Capture" << endl);
		cnt_calibration_frames = 0;
		roiPoints.clear();	
			
		// Print the current view of the camera
		start = clock();
		convertImage(*result_cluster, image, mycloud->width, mycloud->height, 8);
		end = clock();
		cout << "Time required for draw PCL: "<< (double)(end-start)/CLOCKS_PER_SEC << " seconds." << "\n\n";
	    	image.copyTo(im_draw);
		cv::imshow("image",im_draw);
		
		// if user hits 'space'
		if (((char)key) == 32)
		{
			DEBUG(1, cout << "Calibration..." << endl);
			mode = calibration;
		}
	}
	
	if (mode == calibration)
	{
		cnt_calibration_frames++;
		start = clock();
		std::vector<cv::Point3f> mean_shift_points = calcMeanShiftPoints(result_cluster);
		end = clock();
		cout << "Time required for [" << mean_shift_points.size() << "] mean shift points: "<< (double)(end-start)/CLOCKS_PER_SEC << " seconds." << "\n\n";
		std::sort(mean_shift_points.begin(), mean_shift_points.end(), lexico_compare3f);
	    	//mean_shift_points.erase(std::unique(mean_shift_points.begin(), mean_shift_points.end(), points_are_equal3f), mean_shift_points.end());
	    	

		// calculate center of mean shift points
		cv::Point3f center = getCenterPoint(mean_shift_points);
		cv::Point3f max;
	
		// get point far away from the center
		float max_dist = 0;
		for(int i = 0; i < mean_shift_points.size(); i++)
		{
			float dist = sqrt(pow(mean_shift_points[i].x - center.x, 2) + pow(mean_shift_points[i].y - center.y, 2) + pow(mean_shift_points[i].z - center.z, 2));
			if(dist > max_dist)
			{
				max_dist = dist;
				max = mean_shift_points[i];
			}
		}
		
		std::vector<cv::Point2f> projectedPoints;
		if (mean_shift_points.size() > 0)
			cv::projectPoints(mean_shift_points, rvec, tvec, intrinsic, distCoeffs, projectedPoints);
		
		// Print the current view of the camera
		convertImage(*result_cluster, image, mycloud->width, mycloud->height, 8);
	    	image.copyTo(im_draw);
	    	
	    	std::sort(projectedPoints.begin(), projectedPoints.end(), lexico_compare2f);
    		//projectedPoints.erase(std::unique(projectedPoints.begin(), projectedPoints.end(), points_are_equal2f), projectedPoints.end());
	    	
	    	cout << "Mean Shift Particles: " << num_particles << " | current particles: " << projectedPoints.size() << endl;
	    	// draw mean shift circles
	    	int count_circles = 1;
	    	for(int i = 0; i < projectedPoints.size(); i++)
	    	{
	    		if (i + 1 == projectedPoints.size() || !points_are_equal2f(projectedPoints[i], projectedPoints[i+1]))
	    		{
	    			cv::circle(im_draw, projectedPoints[i], 30, cv::Scalar( 255, 0, 0 ), 4);
	    			cv::putText(im_draw,  boost::to_string(count_circles).c_str(), projectedPoints[i]-cv::Point2f(10,0), 
	    				    CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar( 0, 0, 255 ), 3);
	    			count_circles = 1;
	    		}
	    		else
	    		{
	    			count_circles++;
	    		}
	    	}
	    	drawCircle(im_draw, center, 40, cv::Scalar( 0, 255, 0 ), 4); // draw center point
	    	drawCircle(im_draw, max, 40, cv::Scalar( 0, 255, 255 ), 4); // draw point far away from center point
		cv::imshow("image",im_draw);
		
		roiPoints.push_back(max);
		
		if(cnt_calibration_frames >= num_calibration_frames)
		{
	    		// get path
	    		start = clock();
	    		cout << "Start calculating path" << endl;
			std::vector<int> path = getShortestPath(result_cluster, roi_center, center, 0.05);
	    		
	    		// draw path
	    		convertImage(*mycloud, image, mycloud->width, mycloud->height, 8);
			image.copyTo(im_draw);
	    		roi_center = getClusterPoint(roiPoints);
	    		
	    		BOOST_FOREACH(int i, path)
			{
				cv::Point3f p;
				p.x = result_cluster->points[i].x;
				p.y = result_cluster->points[i].y;
				p.z = result_cluster->points[i].z;
				drawCircle(im_draw, p, 10, cv::Scalar( 0, 0, 255 ), 4);
			}
			
			drawCircle(im_draw, roi_center, 20, cv::Scalar( 0, 255, 0 ), 4);
	    		drawCircle(im_draw, center, 20, cv::Scalar( 0, 255, 255 ), 4);
	    		
	    		cv::imshow("path",im_draw);
			cv::namedWindow( "path", CV_WINDOW_AUTOSIZE );
			
			end = clock();
			cout << "Time required for calculating path: "<< (double)(end-start)/CLOCKS_PER_SEC << " seconds." << "\n\n";
	    	
	    		
	    		cout << "Press [SPACE] for start tracking" << endl << "Press [ESC] for exit" << endl << "Press [c] for capturing again" << endl;
			key = cv::waitKey(0);
			
			if (((char)key) == 32)
			{
				mode = tracking;
				DEBUG(1, cout << "Start tracking..." << endl);
			}
			else if (((char)key) == 27)
			{
				mode = stop;
			}
			else
			{
				mode = capture;
			}
			
		}
	}
	
	if (mode == tracking)
	{
		DEBUG(2, cout << "Mode tracking" << endl);
		
		calcMeanShift(roi_center,  result_cluster, (float)mean_shift_radius / 100);
		
		// remove hand
    		start = clock();
    		DEBUG(1, cout << "Start remove hand" << endl);
    		pcl::PointCloud<pcl::PointXYZRGB>::Ptr roi (new pcl::PointCloud<pcl::PointXYZRGB>);
		removeHand(mycloud, roi_center, roi);
		
		end = clock();
		DEBUG(1, cout << "Time required for remove hand: "<< (double)(end-start)/CLOCKS_PER_SEC << " seconds." << "\n\n");
		
		convertImage(*mycloud, image, mycloud->width, mycloud->height, 1);
		image.copyTo(im_draw);
		cv::namedWindow( "roi", CV_WINDOW_AUTOSIZE );
    		cv::imshow("roi",im_draw);
	    		
		
		trackImage(mycloud);
		if (lost_pose_counter >= max_loose_pose)
		{
			mode = stop;
		}
		
	}
	
	if (mode == stop)
	{
		DEBUG(2, cout << "Mode stop" << endl);
		stopCamera = true;
		return;
	}
	
	
	// If user hits 'ESC'
	if (((char)key) == 27)
	{
		stopCamera = true;
	}

	if (((char)key) == 's')
	{
		stringstream stream;
		stream << "inputCloud" << filesSaved << ".pcd";
		string filename = stream.str();
		if (io::savePCDFile(filename, *mycloud, true) == 0)
		{
			filesSaved++;
			cout << "Saved " << filename << "." << endl;
		}
		else PCL_ERROR("Problem saving %s.\n", filename.c_str());
		convertImage(*mycloud, image);
		imwrite( filename + ".jpg", image );
	}
	
	gend = clock();
	cout << "Time required for CB: "<< (double)(gend-gstart)/CLOCKS_PER_SEC << " seconds." << "\n\n";
	
}


/*********************************************************************
 * Calculate the Transformation Matrix for projecting 3D Points to 2D Points
 ********************************************************************/
void calculateCameraSettings(PointCloud<PointXYZRGB>::Ptr& cloud)
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
	
		
	if(solvePnP(objectPoints, imagePoints, intrinsic, distCoeffs, rvec, tvec) )
	{
		DEBUG(1, cout << "Translation matrix calculated" << endl);
		init_camera = true;
	}
	
}

/*********************************************************************
 * This function removes the background of the image
 ********************************************************************/
void removeBackground(PointCloud<PointXYZRGB>::Ptr& cloud, PointCloud<PointXYZRGB>::Ptr& cloud_result)
{
	
	 // Create 2nd Pointcloud for manipulation
	  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZRGB>);
	  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
	  
	  // Remove NaN points from the cloud
	  std::vector<int> indices;
	  pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, indices);

	  // Create the filtering object: downsample the dataset using a leaf size of 1cm
	  pcl::VoxelGrid<pcl::PointXYZRGB> vg;
	  
	  vg.setInputCloud (cloud_filtered);
	  vg.setLeafSize (0.04f, 0.04f, 0.04f);
	  vg.filter (*cloud_filtered);
	  std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*
	  
	  

	  // Create the segmentation object for the planar model and set all the parameters
	  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
	  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGB> ());
	  pcl::PCDWriter writer;
	  seg.setOptimizeCoefficients (true);
	  seg.setModelType (pcl::SACMODEL_PLANE);
	  seg.setMethodType (pcl::SAC_RANSAC);
	  seg.setMaxIterations (70);
	  seg.setDistanceThreshold (0.10);

	  int i=0, nr_points = (int) cloud_filtered->points.size ();
	  while (false && cloud_filtered->points.size () > 0.3 * nr_points)
	  {
	    // Segment the largest planar component from the remaining cloud
	    seg.setInputCloud (cloud_filtered);
	    seg.segment (*inliers, *coefficients);
	    if (inliers->indices.size () == 0)
	    {
	      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
	      break;
	    }

	    // Extract the planar inliers from the input cloud
	    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
	    extract.setInputCloud (cloud_filtered);
	    extract.setIndices (inliers);
	    extract.setNegative (false);

	    // Get the points associated with the planar surface
	    extract.filter (*cloud_plane);
	    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

	    // Remove the planar inliers, extract the rest
	    extract.setNegative (true);
	    extract.filter (*cloud_f);
	    *cloud_filtered = *cloud_f;
	  }

	  // Creating the KdTree object for the search method of the extraction
	  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	  tree->setInputCloud (cloud_filtered);

	  std::vector<pcl::PointIndices> cluster_indices;
	  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	  ec.setClusterTolerance (0.15); // 5cm
	  ec.setMinClusterSize (70);
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
	    double min_dist = DBL_MAX;
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
	    	DEBUG(2, cout << "Center" << i << ": " << dist << endl);
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
	    	cloud_result->height = 1;
	    	cloud_result->is_dense = true;
		
		// Create the filtering object
		pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
		sor.setInputCloud (cloud_result);
		sor.setMeanK (50);
		sor.setStddevMulThresh (1.0);
		sor.filter (*cloud_result);
		
		/*
		std::stringstream ss;
		ss << "./clusters/clusterall_" << filesSaved << "_" << cluster_indices.size() << ".pcd";
	    	writer.write<pcl::PointXYZRGB> (ss.str (), *cloud_cluster_all, false);
		ss.str("");
		ss << "./clusters/cluster_" << filesSaved << ".pcd";
	    	writer.write<pcl::PointXYZRGB> (ss.str (), cloud_clusters[result_cluster_index], false);
	    	filesSaved++;*/
	}
	
	

	/*
	BOOST_FOREACH (pcl::PointXYZRGB& pt, cloud->points)
	{
		if (pt.z != pt.z || pt.x != pt.x || pt.y != pt.y)
		{
			pt.r = 0;
			pt.g = 0;
			pt.b = 0;
		
		}
			
		if(pt.z > float(max_camera_distance) / 100)
		{
			pt.x = bad_point;
			pt.y = bad_point;
			pt.z = bad_point;
			pt.r = 0;
			pt.g = 0;
			pt.b = 0;
		}
		//printf ("\t(%f, %f, %f, %d, %d, %d)\n", pt.x, pt.y, pt.z, pt.r, pt.g, pt.b);
		
	}
	*/ 


	
}


// remove hand
void removeHand(PointCloud<PointXYZRGB>::Ptr& cloud, cv::Point3f roi_center, PointCloud<PointXYZRGB>::Ptr& result )
{
	pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;

	kdtree.setInputCloud (cloud);
	std::vector<int> indices = getNearestNeigboursInRadius(kdtree, 0.2, roi_center);
	
	std::sort (indices.begin(), indices.end());
	
	//result->points.clear();
	for (int i = 0; i < cloud->points.size(); i++)
	{
		if (!(std::binary_search (indices.begin(), indices.end(), i)))
		{
			cloud->points[i].x = bad_point;
			cloud->points[i].y = bad_point;
			cloud->points[i].z = bad_point;
			cloud->points[i].r = 0;
			cloud->points[i].g = 0;
			cloud->points[i].b = 0;
		}
		
		
	
	}
	result->width = result->points.size ();
    	result->height = 1;
    	result->is_dense = false;
}


// Calc defined numbers of mean shift points over cloud
std::vector<cv::Point3f> calcMeanShiftPoints(PointCloud<PointXYZRGB>::Ptr& cloud)
{
	std::vector<cv::Point3f> result;
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
		calcMeanShift(result[i],  cloud, (float)mean_shift_radius / 100);
	}
	
	return result;
}

// Calc mean shift
void calcMeanShift(cv::Point3f &p, PointCloud<PointXYZRGB>::Ptr& cloud, float radius)
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

// calculate the center of several points
cv::Point3f getCenterPoint(std::vector<cv::Point3f> points)
{
	// calculate center of mean shift points
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tmp (new pcl::PointCloud<pcl::PointXYZRGB>);
	for(int i = 0; i < points.size(); i++)
	{
		pcl::PointXYZRGB p;
		p.x = points[i].x;
		p.y = points[i].y;
		p.z = points[i].z;
		cloud_tmp->points.push_back(p);
	}

	cloud_tmp->width = cloud_tmp->points.size ();
	cloud_tmp->height = 1;
	cloud_tmp->is_dense = true;
	Eigen::Matrix<float, 4, 1 > tmp;
	pcl::compute3DCentroid(*cloud_tmp , tmp);
	cv::Point3f center(tmp[0], tmp[1], tmp[2]);
	return center;
}

// get the center of the biggest cluster
cv::Point3f getClusterPoint(std::vector<cv::Point3f> points)
{
	/*std::vector<int> labels;
	std::vector<cv::Point3f> centers;
	
	cv::kmeans(points, 2, labels, cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
	return centers[0];*/
	
	//cv::Point3f clusters[points.size()][points.size()];
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
				
				if(dist < 0.1)
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
			drawCircle(image, clusters[i][j], 20, cv::Scalar( b, g, r ), 4);
		}
		
		if(clusters[i].size() > max_size)
		{
			
			max_size = clusters[i].size();
			std::vector<cv::Point3f> p = clusters[i];
			result = p[p.size() - 1];
		}
	}
	cv::namedWindow( "calibration", CV_WINDOW_AUTOSIZE );
	cv::imshow("calibration",image);
	
	return result;
}

std::vector<int> getKNearestNeigbours(pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree, int K, cv::Point3f p)
{
	pcl::PointXYZRGB searchPoint;

	searchPoint.x = p.x;
	searchPoint.y = p.y;
	searchPoint.z = p.z;

	// K nearest neighbor search

	std::vector<int> pointIdxNKNSearch(K);
	std::vector<float> pointNKNSquaredDistance(K);

	
	kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
	
	return pointIdxNKNSearch;
}

std::vector<int> getNearestNeigboursInRadius(pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree, float radius, cv::Point3f p)
{
	pcl::PointXYZRGB searchPoint;

	searchPoint.x = p.x;
	searchPoint.y = p.y;
	searchPoint.z = p.z;

	// K nearest neighbor search

	std::vector<int> pointIdxNKNSearch;
	std::vector<float> pointNKNSquaredDistance;

	
	kdtree.radiusSearch (searchPoint, radius, pointIdxNKNSearch, pointNKNSquaredDistance);
	
	
	
	return pointIdxNKNSearch;
}


class Node {
  public:
    float x,y,z;
    int pcl_index;
    int pcl_pred_index; //predecessor
    float gscore;
    float fscore;
  
    void init(PointXYZRGB p, int index, int pred_index);
    float calcDistance(Node n);
    cv::Point3f toPoint();
    
    bool operator < (const Node& n) const
    {
        return (pcl_index < n.pcl_index);
    }
    
    bool operator == (const Node& n) const
    {
        return (pcl_index == n.pcl_index);
    }
};

float Node::calcDistance (Node n) {
  return sqrt( pow(x-n.x, 2) + pow(y-n.y, 2) + pow(z-n.z, 2) );
}

cv::Point3f Node::toPoint()
{
	cv::Point3f p;
	p.x = x;
	p.y = y;
	p.z = z;
	return p;
}

void Node::init(PointXYZRGB p, int index, int pred_index)
{
	x = p.x;
	y = p.y;
	z = p.z;
	pcl_index = index;
	pcl_pred_index = pred_index;
	gscore = std::numeric_limits<float>::infinity();
}

int getMinNodeIndex(std::vector<Node> list)
{
	int index = -1;
	for(int i = 0; i < list.size(); i++)
	{
		if (index == -1)
			index = i;
				
		else if(list[i].fscore < list[index].fscore)
			index = i;
	}
	
	return index;

}

std::vector<int> getShortestPath(PointCloud<PointXYZRGB>::Ptr& cloud, cv::Point3f start, cv::Point3f end, float maxDist)
{
	pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;

	kdtree.setInputCloud (cloud);
	
	// get start point
	std::vector<int> vstart = getKNearestNeigbours(kdtree, 1, start);
	std::vector<int> vend = getKNearestNeigbours(kdtree, 1, end);
	
	std::vector<int> result;
	
	if(vstart.size() == 0 || vend.size() == 0)
	{
		cout << "getShortestPath: Error getting start or end point" << endl;
		return result;
	}
	
	PointXYZRGB p1 = cloud->points[vstart[0]];
	PointXYZRGB p2 = cloud->points[vend[0]];
	
	
	std::vector<Node> openSet;
	std::vector<Node> closedSet;
	
	
	
	Node s;
	s.init(p1, vstart[0], -1);
	s.gscore = 0;
	
	Node e;
	e.init(p2, vend[0], -1);
	e.fscore = 0;
	
	s.fscore = s.calcDistance(e);
	
	
	// add start node
	openSet.push_back(s);
	
	
	/*convertImage(*cloud, image, mycloud->width, mycloud->height, 8);
	image.copyTo(im_draw);
	drawCircle(im_draw, s.toPoint(), 40, cv::Scalar( 0, 255, 0 ), 4);
	drawCircle(im_draw, e.toPoint(), 40, cv::Scalar( 0, 255, 255 ), 4);
	
	
	cv::imshow("path",im_draw);
	cv::namedWindow( "path", CV_WINDOW_AUTOSIZE );*/
	
	bool found = false;
	while(openSet.size() > 0)
	{
		//cout << "OpenSet.size() = " << openSet.size() << endl;
		int min_index = getMinNodeIndex(openSet);
		Node min = openSet[min_index];
		//drawCircle(im_draw, min.toPoint(), 10, cv::Scalar( 0, 0, 255 ), 4);
		//cv::imshow("path",im_draw);
		//cv::waitKey(0);
		if(min == e)
		{
			e.pcl_pred_index = min.pcl_pred_index;
			found = true;
			break;
		}
		
		closedSet.push_back(min);
		std::sort (closedSet.begin(), closedSet.end());
		openSet.erase(openSet.begin() + min_index);
		cv::Point3f current;
		current.x = min.x;
		current.y = min.y;
		current.z = min.z;
		cout << "Current: " << current << endl;
		std::vector<int> neighbours_index = getNearestNeigboursInRadius(kdtree, maxDist, current);
		//std::vector<int> neighbours_index = getKNearestNeigbours(kdtree, 9, current);
		cout << "Found neighbours:" << neighbours_index.size() << endl;
		

		BOOST_FOREACH(int i, neighbours_index)
		{
			PointXYZRGB p = cloud->points[i];
			Node node_neighbour;
			node_neighbour.init(p, i, min.pcl_index);
			float dist = min.calcDistance(node_neighbour);
			
			//drawCircle(im_draw, node_neighbour.toPoint(), 10, cv::Scalar( 255, 0, 0 ), 4);
			
			if (dist > maxDist)
				continue;
			
			if (std::find(closedSet.begin(), closedSet.end(), node_neighbour) != closedSet.end())
			{
				continue;
			}
			
			float gscore = min.gscore +  dist;
			
			if(gscore >= node_neighbour.gscore)
				continue;
			
			
			
			node_neighbour.gscore = gscore;
			node_neighbour.fscore = gscore + node_neighbour.calcDistance(e);
			
			
			// discoverd new node
			if(std::find(openSet.begin(), openSet.end(), node_neighbour) == openSet.end())
			{
				openSet.push_back(node_neighbour);
				std::sort (openSet.begin(), openSet.end());
			}
		}
		
	}
	
	cout << "A* FINISHED" << endl;
	if (found)
	{
		int index = e.pcl_pred_index;
		cout << "Index " << e.pcl_pred_index << endl;
		
		while(index >= 0)
		{
			bool node_found = false;
			for(int i = 0; i < closedSet.size(); i++)
			{
				if (closedSet[i].pcl_index == index)
				{
					result.push_back(index);
					index = closedSet[i].pcl_pred_index;
					node_found = true;
					break;
					
				}
			
			}
			
			if(!node_found)
			{
				cout << "A* Search error: Not not found " << index << endl;
			}
		
		}
	}
	else
	{
		cout << "A* ERROR: No path found" << endl;
		
		BOOST_FOREACH(Node n, closedSet)
		{
			if(n.pcl_index >= 0)
				result.push_back(n.pcl_index);
		}
	}
	cout << "FOUND PATH: " << result.size() << endl;
	return result;
}



/*********************************************************************
 * This function removes the hand from the image
 ********************************************************************/
void postProcessing(PointCloud<PointXYZRGB>::Ptr& cloud)
{
	pcl::search::Search <pcl::PointXYZRGB>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGB> > (new pcl::search::KdTree<pcl::PointXYZRGB>);
	pcl::IndicesPtr indices (new std::vector <int>);
	pcl::PassThrough<pcl::PointXYZRGB> pass;
	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (0.0, 1.0);
	pass.filter (*indices);

	pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
	reg.setInputCloud (cloud);
	reg.setIndices (indices);
	reg.setSearchMethod (tree);
	reg.setDistanceThreshold (8);
	reg.setPointColorThreshold (6);
	reg.setRegionColorThreshold (5);
	reg.setMinClusterSize (300);

	std::vector <pcl::PointIndices> clusters;
	std::vector <pcl::PointIndices> del_clusters;
	reg.extract (clusters);

	DEBUG(2, std::cout << "Extracted clusters" << std::endl);

	uint8_t min_r = 72;
	uint8_t min_g = 60;
	uint8_t min_b = 30;

	uint8_t max_r = 138;
	uint8_t max_g = 110;
	uint8_t max_b = 99;


	std::vector<int> del_cluster;
	for(int i = 0; i < clusters.size(); i++)
	{
		std::vector<uint8_t> r;
		std::vector<uint8_t> g;
		std::vector<uint8_t> b;

		for (int counter = 0; counter < clusters[i].indices.size(); counter++)
		{
			int index = clusters[i].indices[counter];
			r.push_back(cloud->points[index].r);
			g.push_back(cloud->points[index].g);
			b.push_back(cloud->points[index].b);
	
		}

		std::sort(r.begin(), r.end());
		std::sort(g.begin(), g.end());
		std::sort(b.begin(), b.end());

		uint8_t med_r = r[r.size()/2];
		uint8_t med_g = g[r.size()/2];
		uint8_t med_b = b[r.size()/2];

		DEBUG(1, cout << "Found Cluster: " << int(med_r) << "|" << int(med_g) << "|" << int (med_b) << endl);

		if (min_r <= med_r && med_r <= max_r && 
		    min_g <= med_g && med_g <= max_g && 
		    min_b <= med_b && med_b <= max_b   )
		{
			del_cluster.push_back(i);
			del_clusters.push_back(clusters[i]);
			DEBUG(2, cout << "Del Cluster: " << int(med_r) << "|" << int(med_g) << "|" << int (med_b) << endl);
		} 
	}
	
	pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
	io::savePCDFile("cloud_col.pcd", *colored_cloud, true);

	for (int i = 0; i < del_cluster.size(); i++)
	{
		// Extract the hand  from the input cloud
		pcl::ExtractIndices<pcl::PointXYZRGB> extract;
		extract.setInputCloud (cloud);
		pcl::PointIndices tmp = clusters[del_cluster[i]];
		 pcl::PointIndices::Ptr inliers = boost::make_shared<pcl::PointIndices>(tmp);
		extract.setIndices (inliers);
		extract.setNegative (true);

		// Get the points associated with the planar surface
		extract.filter (*cloud);
	}	
	
	
	 

}


/*********************************************************************
 * This function tracks one image
 ********************************************************************/
void trackImage(const PointCloud<PointXYZRGB>::ConstPtr& cloud)
{
	cout<<"---------------- FRAME #"<<i<<" -----------------------"<<endl;

	convertImage(*cloud, image);
	image.copyTo(im_draw);

	tsf.setDebugImage(im_draw);

	// track
	{ pcl::ScopeTime t("overall time");
	have_pose = tsf.track(*cloud, i, pose, conf_ransac_iter, conf_tracked_points);
	mytime = t.getTime();
	} //-- overall time --

	// ---- END batch filtering ---

	DEBUG(1, cout<<"conf (ransac, tracked points): "<<conf_ransac_iter<<", "<<conf_tracked_points<<endl);
	if (!have_pose) 
	{
		lost_pose_counter++;
		cout<<"############################ Lost pose: " << lost_pose_counter << "  #############################" << endl;
	}
	else
	{
		lost_pose_counter = 0;
	}
	all_poses.push_back(std::make_pair(pose,-1));
	v4r::invPose(pose, inv_pose);

	// get filtered frame
	tsf.getFilteredCloudNormals(filt_cloud, filt_pose, timestamp);

	mean_time += mytime;
	cnt_time++;
	DEBUG(1, cout<<"mean="<<mean_time/double(cnt_time)<<"ms ("<<1000./(mean_time/double(cnt_time))<<"fps)"<<endl);
	DEBUG(1, cout<<"timestamp (c/f): "<<i<<"/"<<timestamp<<endl);

	// debug out draw
	if (display)
	{
		drawConfidenceBar(im_draw, conf_ransac_iter, 50, 200, 30);
		drawConfidenceBar(im_draw, conf_tracked_points, 50, 200, 50);
		cv::imshow("image",im_draw);
		
		//if (conf_ransac_iter<0.2) cv::waitKey(0); // Not sure what this do
	}
	else usleep(50000);
	
	// Increment index for tracking
	i++;
}


/*********************************************************************
 * This function intializes the V4R tracker
 ********************************************************************/
void initTracker()
{
	DEBUG(1, cout << "Init Tracker..." << endl);
	mean_time = 0;
	cnt_time = 0;
	lost_pose_counter = 0;

	// configure camera tracking, temporal smothing and mapping
	tsf.setCameraParameter(intrinsic);
	tsf.setCameraParameterTSF(intrinsic, 640, 480);

	v4r::TSFVisualSLAM::Parameter param;
	param.map_param.refine_plk = true;
	param.map_param.detect_loops = true;
	param.map_param.ba.depth_error_scale = 100;
	param.filt_param.batch_size_clouds = 20;
	param.diff_cam_distance_map = 0.2;
	param.diff_delta_angle_map = 3;
	param.filt_param.type = 3;  //0...ori. col., 1..col mean, 2..bilin., 3..bilin col and depth with cut off thr
	param.map_param.ba.optimize_delta_cloud_rgb_pose_global = false;
	param.map_param.ba.optimize_delta_cloud_rgb_pose = false;
	tsf.setParameter(param);

	v4r::FeatureDetector::Ptr detector(new v4r::FeatureDetector_KD_FAST_IMGD());
	tsf.setDetectors(detector, detector);


	conf_ransac_iter = 1;
	conf_tracked_points = 1;
	ts_last=0;
	i = 0;
}

/*********************************************************************
 * This function stops the tracker and writes the pcd file to disk
 ********************************************************************/
void stopTracker()
{
	DEBUG(1, cout << "Stop tracking..." << endl);
	// optimize map
	tsf.stop();
	tsf.optimizeMap();

	//tsf.getMap();
	tsf.getCameraParameter(intrinsic_opti, dist_coeffs_opti);

	// create model in global coordinates
	cout<<"Create pointcloud model..."<<endl;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr glob_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::PolygonMesh mesh;
	v4r::TSFGlobalCloudFilteringSimple gfilt;
	v4r::TSFGlobalCloudFilteringSimple::Parameter filt_param;
	filt_param.filter_largest_cluster = false;
	filt_param.voxel_size = voxel_size;
	filt_param.thr_weight = thr_weight;
	filt_param.thr_delta_angle = thr_delta_angle;
	filt_param.poisson_depth = poisson_depth;
	gfilt.setParameter(filt_param);

	gfilt.setCameraParameter(intrinsic_opti, dist_coeffs_opti);
	gfilt.getGlobalCloudFiltered(tsf.getMap(), *glob_cloud);
	
	pcl::copyPointCloud<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>(*glob_cloud, *mycloud);
	//postProcessing(mycloud);

	if (file_cloud.size()>0) pcl::io::savePCDFileBinary(file_cloud, *mycloud);

	cout<<"Creat mesh..."<<endl;
	gfilt.getMesh(glob_cloud, mesh);

	// store resulting files
	if (file_mesh.size()>0) pcl::io::savePLYFile(file_mesh, mesh);
}


/*********************************************************************
 * This function converts a cloud image to an cv::Mat
 ********************************************************************/
void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &_cloud, cv::Mat &_image)
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
void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &_image, int width, int height, int radius = 1)
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
	
	std::vector<cv::Point2f> projectedPoints;
	cv::projectPoints(objectPoints, rvec, tvec, intrinsic, distCoeffs, projectedPoints);
	
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
void convertImage(const pcl::PointCloud<pcl::PointXYZRGBA> &_cloud, cv::Mat &_image)
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
void convertImage(const pcl::PointCloud<pcl::PointXYZRGBNormal> &_cloud, cv::Mat &_image)
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


/*********************************************************************
 * This functions draws the coordinate system
 ********************************************************************/
void drawCoordinateSystem(cv::Mat &im, const Eigen::Matrix4f &_pose, const cv::Mat_<double> &_intrinsic, const cv::Mat_<double> &dist_coeffs, double size, int thickness)
{
  Eigen::Matrix3f R = _pose.topLeftCorner<3,3>();
  Eigen::Vector3f t = _pose.block<3, 1>(0,3);

  Eigen::Vector3f pt0 = R * Eigen::Vector3f(0,0,0) + t;
  Eigen::Vector3f pt_x = R * Eigen::Vector3f(size,0,0) + t;
  Eigen::Vector3f pt_y = R * Eigen::Vector3f(0,size,0) + t;
  Eigen::Vector3f pt_z = R * Eigen::Vector3f(0,0,size) +t ;

  cv::Point2f im_pt0, im_pt_x, im_pt_y, im_pt_z;

  if (!dist_coeffs.empty())
  {
    v4r::projectPointToImage(&pt0[0], &_intrinsic(0), &dist_coeffs(0), &im_pt0.x);
    v4r::projectPointToImage(&pt_x[0], &_intrinsic(0), &dist_coeffs(0), &im_pt_x.x);
    v4r::projectPointToImage(&pt_y[0], &_intrinsic(0), &dist_coeffs(0), &im_pt_y.x);
    v4r::projectPointToImage(&pt_z[0], &_intrinsic(0), &dist_coeffs(0), &im_pt_z.x);
  }
  else
  {
    v4r::projectPointToImage(&pt0[0], &_intrinsic(0), &im_pt0.x);
    v4r::projectPointToImage(&pt_x[0], &_intrinsic(0), &im_pt_x.x);
    v4r::projectPointToImage(&pt_y[0], &_intrinsic(0), &im_pt_y.x);
    v4r::projectPointToImage(&pt_z[0], &_intrinsic(0), &im_pt_z.x);
  }

  cv::line(im, im_pt0, im_pt_x, CV_RGB(255,0,0), thickness);
  cv::line(im, im_pt0, im_pt_y, CV_RGB(0,255,0), thickness);
  cv::line(im, im_pt0, im_pt_z, CV_RGB(0,0,255), thickness);
}

/*********************************************************************
 * This functions draws the confidence bar in the viewer
 ********************************************************************/
void drawConfidenceBar(cv::Mat &im, const double &conf, int x_start, int x_end, int y)
{
  int bar_start = x_start, bar_end = x_end;
  int diff = bar_end-bar_start;
  int draw_end = diff*conf;
  double col_scale = (diff>0?255./(double)diff:255.);
  cv::Point2f pt1(0,y);
  cv::Point2f pt2(0,y);
  cv::Vec3b col(0,0,0);

  if (draw_end<=0) draw_end = 1;

  for (int i=0; i<draw_end; i++)
  {
    col = cv::Vec3b(255-(i*col_scale), i*col_scale, 0);
    pt1.x = bar_start+i;
    pt2.x = bar_start+i+1;
    cv::line(im, pt1, pt2, CV_RGB(col[0],col[1],col[2]), 8);
  }
}

// draw 3D Point on img
void drawCircle(cv::Mat &img, cv::Point3f center, int radius, cv::Scalar color, int thickness)
{
	std::vector<cv::Point3f> tmpPoints;
	std::vector<cv::Point2f> centerPoint;
	tmpPoints.push_back(center);
	cv::projectPoints(tmpPoints, rvec, tvec, intrinsic, distCoeffs, centerPoint);
    	cv::circle(img, centerPoint[0], radius, color, thickness);
}


// Lexicographic compare, same as for ordering words in a dictionnary:
// test first 'letter of the word' (x coordinate), if same, test 
// second 'letter' (y coordinate).
bool lexico_compare2f(const cv::Point2f& p1, const cv::Point2f& p2) {
    if(p1.x < p2.x) { return true; }
    if(p1.x > p2.x) { return false; }
    return (p1.y < p2.y);
}


 bool points_are_equal2f(const cv::Point2f& p1, const cv::Point2f& p2) {
   return ((p1.x == p2.x) && (p1.y == p2.y));
 }
 
bool lexico_compare3f(const cv::Point3f& p1, const cv::Point3f& p2) {
    if(p1.x < p2.x) { return true; }
    if(p1.x > p2.x) { return false; }
    if(p1.y < p2.y) { return true; }
    if(p1.y > p2.y) { return false; }
    return (p1.z < p2.z);
}


bool points_are_equal3f(const cv::Point3f& p1, const cv::Point3f& p2) {
   return ((p1.x == p2.x) && (p1.y == p2.y) && (p1.z == p2.z));
}




