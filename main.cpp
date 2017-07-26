// Modeling Tool for in-hand-objects
// written by Dominik Streicher
// based on Hannes Prankl camera_tracking_and_mapping

#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>

#include <iostream>

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




/* ######################### Methods ############################## */
void printUsage(const char*);
void grabberCallback(const PointCloud<PointXYZRGB>::ConstPtr&);

void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &image);
void convertImage(const pcl::PointCloud<pcl::PointXYZRGBA> &_cloud, cv::Mat &_image);
void convertImage(const pcl::PointCloud<pcl::PointXYZRGBNormal> &_cloud, cv::Mat &_image);

void drawCoordinateSystem(cv::Mat &im, const Eigen::Matrix4f &pose, const cv::Mat_<double> &intrinsic, const cv::Mat_<double> &dist_coeffs, double size, int thickness);
void drawConfidenceBar(cv::Mat &im, const double &conf, int x_start=50, int x_end=200, int y=30);

void initTracker();
void stopTracker();
void trackImage(const PointCloud<PointXYZRGB>::ConstPtr&);



/* ######################### Variables ############################# */
PointCloud<PointXYZRGB>::Ptr cloudptr(new PointCloud<PointXYZRGB>); 	// A cloud that will store color info.
Grabber* openniGrabber;                                               	// OpenNI grabber that takes data from the device.
unsigned int filesSaved = 0;                                          	// For the numbering of the clouds saved to disk.
bool stopCamera(false);							// Stop the camera callback
enum Mode { capture, tracking };					// Current mode
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

cv::Mat_<double> distCoeffs;// = cv::Mat::zeros(4, 1, CV_64F);
cv::Mat_<double> intrinsic = cv::Mat_<double>::eye(3,3);
cv::Mat_<double> dist_coeffs_opti = cv::Mat::zeros(4, 1, CV_64F);
cv::Mat_<double> intrinsic_opti = cv::Mat_<double>::eye(3,3);
cv::Mat_<double> intrinsic_tsf = cv::Mat_<double>::eye(3,3);

Eigen::Matrix4f pose;
float voxel_size = 0.0005;
double thr_weight = 2;      //e.g. 10    // surfel threshold for the final model
double thr_delta_angle = 80; // e.g. 80
int poisson_depth = 6;
int display = true;

cv::Point track_win[2];


/*********************************************************************
 * Main entrypoint of the program
 ********************************************************************/
int main(int argc, char** argv)
{
	if (console::find_argument(argc, argv, "-h") >= 0)
	{
		printUsage(argv[0]);
		return -1;
	}


	openniGrabber = new OpenNIGrabber();
	if (openniGrabber == 0)
		return -1;
	boost::function<void (const PointCloud<PointXYZRGB>::ConstPtr&)> f =
		boost::bind(&grabberCallback, _1);
	openniGrabber->registerCallback(f);
	
	// gui for the visualization
	cv::namedWindow( "image", CV_WINDOW_AUTOSIZE );
	
	// initialize tracker
	initTracker();
 	
	// start callback for the camera images
	openniGrabber->start();

	while(!stopCamera)
		boost::this_thread::sleep(boost::posix_time::seconds(1));

	// stop the camera 
	openniGrabber->stop();
	
	// stop the tracker
	if (mode == tracking)
		stopTracker();
	
	
	cout<<"Finished!"<<endl;


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
void grabberCallback(const PointCloud<PointXYZRGB>::ConstPtr& cloud)
{
	int key= cv::waitKey(100);
	if (mode == capture)
	{
		// Print the current view of the camera
		convertImage(*cloud, image);
	    	image.copyTo(im_draw);
		cv::imshow("image",im_draw);
		
	
		// if user hits 'space'
		if (((char)key) == 32)
		{
			cout << "Start Tracking..." << endl;
			mode = tracking;
		}
	}
	
	if (mode == tracking)
	{
		trackImage(cloud);
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
		if (io::savePCDFile(filename, *cloud, true) == 0)
		{
			filesSaved++;
			cout << "Saved " << filename << "." << endl;
		}
		else PCL_ERROR("Problem saving %s.\n", filename.c_str());
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

	cout<<"conf (ransac, tracked points): "<<conf_ransac_iter<<", "<<conf_tracked_points<<endl;
	if (!have_pose) cout<<"Lost pose!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;

	all_poses.push_back(std::make_pair(pose,-1));
	v4r::invPose(pose, inv_pose);

	// get filtered frame
	tsf.getFilteredCloudNormals(filt_cloud, filt_pose, timestamp);

	mean_time += mytime;
	cnt_time++;
	cout<<"mean="<<mean_time/double(cnt_time)<<"ms ("<<1000./(mean_time/double(cnt_time))<<"fps)"<<endl;
	cout<<"timestamp (c/f): "<<i<<"/"<<timestamp<<endl;

	// debug out draw
	int key=0;
	if (display)
	{
		drawConfidenceBar(im_draw, conf_ransac_iter, 50, 200, 30);
		drawConfidenceBar(im_draw, conf_tracked_points, 50, 200, 50);
		cv::imshow("image",im_draw);
		if (conf_ransac_iter<0.2) cv::waitKey(0);
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
	cout << "Init Tracker..." << endl;
	mean_time=0;
	cnt_time=0;

	intrinsic(0,0)=intrinsic(1,1)=525;
	intrinsic(0,2)=320, intrinsic(1,2)=240;

	// configure camera tracking, temporal smothing and mapping
	tsf.setCameraParameter(intrinsic);
	intrinsic.copyTo(intrinsic_tsf);
	intrinsic_tsf(0,0) = intrinsic_tsf(1,1) = 840;
	intrinsic_tsf(0,2) = 512; intrinsic_tsf(1,2) = 384;
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
	cout << "Stop tracking..." << endl;
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

	if (file_cloud.size()>0) pcl::io::savePCDFileBinary(file_cloud, *glob_cloud);

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





