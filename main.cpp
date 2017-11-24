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

#include "boost/filesystem.hpp"
#include "FaceDetection.h"
#include "ObjSegmentation.h"
#include "PclManipulation.h"
#include "Camera.h"
#include "SkinDetection.h"

#include <cv.h>
#include <highgui.h>

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


using namespace boost::filesystem;
using namespace std;
using namespace pcl;


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


cv::Point3f getPOI(PointCloud<PointXYZRGB>::Ptr&);




void drawCoordinateSystem(cv::Mat &im, const Eigen::Matrix4f &pose, const cv::Mat_<double> &intrinsic, const cv::Mat_<double> &dist_coeffs, double size, int thickness);
void drawConfidenceBar(cv::Mat &im, const double &conf, int x_start=50, int x_end=200, int y=30);


static void onMouse(int event, int x, int y, int f, void *);

bool lexico_compare2f(const cv::Point2f& p1, const cv::Point2f& p2);
bool points_are_equal2f(const cv::Point2f& p1, const cv::Point2f& p2);
bool lexico_compare3f(const cv::Point3f& p1, const cv::Point3f& p2);
bool points_are_equal3f(const cv::Point3f& p1, const cv::Point3f& p2);

void initTracker();
void stopTracker();
void trackImage(const PointCloud<PointXYZRGB>::ConstPtr&);


/* ######################### Constants ############################# */
const float bad_point = std::numeric_limits<float>::quiet_NaN();


/* ######################### Variables ############################# */
PointCloud<PointXYZRGB>::Ptr mycloud (new PointCloud<PointXYZRGB>); 	// A cloud that will store color info.
Grabber* openniGrabber;                                               	// OpenNI grabber that takes data from the device.
unsigned int filesSaved = 0;                                          	// For the numbering of the clouds saved to disk.
bool stopCamera(false);							// Stop the camera callback
enum Mode { capture, calibration, parameters, tracking, stop };					// Current mode
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
cv::Mat_<cv::Vec3b> im_cb;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

std::string cam_file, filenames;
std::string file_mesh = "mesh.ply";
std::string file_cloud = "cloud.pcd";

cv::Mat_<double> distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
cv::Mat_<double> intrinsic = cv::Mat_<double>::eye(3,3);
cv::Mat_<double> dist_coeffs_opti = cv::Mat::zeros(4, 1, CV_64F);
cv::Mat_<double> intrinsic_opti = cv::Mat_<double>::eye(3,3);

Eigen::Matrix4f pose;
float voxel_size = 0.0001;//0.0005;
double thr_weight = 2;      //e.g. 10    // surfel threshold for the final model
double thr_delta_angle = 80; // e.g. 80
int poisson_depth = 6;
int display = true;

int min_camera_distance = 50; // in [cm]
int num_particles = 20;
int num_calibration_frames = 1;
int cnt_calibration_frames;

cv::Point track_win[2];

int mean_shift_radius = 15; // in cm
cv::Point3f roi_center;
std::vector<cv::Point3f> roiPoints;
cv::Point3f min_point; // nearest point to camera

// camera parameter
Camera cam;
SkinDetection skin_detection;
FaceDetection face;

bool view_error = false;
bool filediskmode = false;
int filedisk_counter = 0;
string read_path;

// min cut parameters
int border_dist=3;
int border_offset=3;

// hysterese params
int hyst_dist=5;

// skin params
int h_range = 3;
int s_range = 10;
int v_range = 10;

// Cluster params
int useGrabCut = 0;
int track_image = 0;

int max_loose_pose = 5; 						// Maximum number of lost poses until the grabber ends
int frame_counter = 0;              // Counts the numbers of tracking frames
int minBinSize = 10;                // Minimum number of entries in a bin of the histogram
int minBinSizeLum = 10;
int sigmaScale = 20;
std::vector<cv::Vec3b> skin_points;
cv::Point3f obj_point(0, 0, 0);

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
	
	if (argc > 1)
	{
		filediskmode = true;
		DEBUG(1, cout << "Enter filedisk mode" << endl);
		read_path = read_path.assign(argv[1]);
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
	cam = Camera(distCoeffs, intrinsic);

  // initialise skin detector
  //skin_detection = SkinDetection();

	// initialize tracker
	initTracker();
 	
	// start callback for the camera images
	openniGrabber->start();
	
	// trackbars
	cv::namedWindow("trackbars",CV_WINDOW_KEEPRATIO);

  //cv::createTrackbar("Max camera distance [cm]","trackbars", &min_camera_distance, 400);
	//cv::createTrackbar("Mean_shift radius [cm]","trackbars", &mean_shift_radius, 100);
	//cv::createTrackbar("Number of patricles","trackbars", &num_particles, 200);
	//cv::createTrackbar("Number of calibraion frames","trackbars", &num_calibration_frames, 1000);
	
	cv::createTrackbar("h range", "trackbars", &h_range, 180);
	cv::createTrackbar("s range", "trackbars", &v_range, 255);
	cv::createTrackbar("v range", "trackbars", &s_range, 255);
	cv::createTrackbar("Hysterese dist", "trackbars", &hyst_dist, 15);
	cv::createTrackbar("Use grabCut", "trackbars", &useGrabCut, 1);
	cv::createTrackbar("Max loose Pose", "trackbars", &max_loose_pose, 10);
	cv::createTrackbar("TrackImage", "trackbars", &track_image, 1);
	cv::createTrackbar("MinBinSize", "trackbars", &minBinSize, 300);
	cv::createTrackbar("MinBinSizeLuminance", "trackbars", &minBinSizeLum, 300);
	cv::createTrackbar("Sigma Scale", "trackbars", &sigmaScale, 50);
	
  cv::namedWindow("orig 2D");
  setMouseCallback("orig 2D", onMouse, 0);

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


static void onMouse(int event, int x, int y, int f, void *)
{

  Mat image = im_cb.clone();
  Vec3b pix = image.at<Vec3b>(y,x);

  int B = pix.val[0];
  int G = pix.val[1];
  int R = pix.val[2];

  int rgb_sum = R + G + B;
  float norm_R = (float) R / rgb_sum;
  float norm_G = (float) G / rgb_sum;

  stringstream s;
  s << "rgR=(" << norm_R << ", " << norm_G << ", " << R << ") at (y,x): (" << y << "," << x << ")";
  putText(image, s.str().c_str(), Point(17,15), FONT_HERSHEY_SIMPLEX, .6, Scalar(0,255,0), 1);

  imshow("orig 2D", image);
}



/*********************************************************************
 * This function is called every time the device has new data.
 ********************************************************************/
void grabberCallback(const PointCloud<PointXYZRGBA>::ConstPtr& cloud)
{

	clock_t gstart, gend, start, end;
	gstart = clock();
	DEBUG(2, cout << "Callback..." << endl);
	// copy to the pcl for write-access
	pcl::copyPointCloud<pcl::PointXYZRGBA, pcl::PointXYZRGB>(*cloud, *mycloud);
	
	if (filediskmode)
	{
		stringstream pcd_file;
		//pcd_file << read_path << "/inputCloud" << filedisk_counter << ".pcd";
		//pcd_file << "./binary_3/inputCloud5.pcd";
		pcd_file << read_path;
		if ( pcl::io::loadPCDFile <pcl::PointXYZRGB> (pcd_file.str(), *mycloud) == -1 )
		{
			std::cout << "Cloud reading failed." << std::endl;
			stopCamera = true;
			return;
		}
	}
	
	if(!cam.init_camera)
	{
		 cam.calculateCameraSettings(mycloud);
	}
	
	
	

  cam.convertImage(*mycloud, image, mycloud->width, mycloud->height, 0);
  image.copyTo(im_draw);
  cv::imshow("orig",im_draw);

  cam.convertImage(*mycloud, image);
  image.copyTo(im_cb);
  //cv::imshow("orig 2D", im_cb);

  int key= cv::waitKey(10);
  if (mode == capture)
  {
    DEBUG(2, cout << "Mode Capture" << endl);
    face.setImage(image);
    if(face.detectFace(false))
    {
      face.showResult();
    }
    else
    {
      cout << "Unable to detect face" << endl;
    }

    // if user hits 'space'
    if (((char)key) == 32)
    {
      DEBUG(1, cout << "Calibration..." << endl);
      mode = calibration;
    }
  }

  if (mode == calibration)
  {
    // Detect face for tracking
    face.setImage(image);
    if (face.detectFace(false))
    {
      
      face.showResult();
      mode = parameters;     
    }
    else
    {
      cout << "Unable to detect face" << endl;
    }
  }


  if (mode == parameters)
  {

    cout << "PRESS ENTER TO SUBMIT CHANGES" << endl;
    skin_detection.minBinSize = minBinSize;
    skin_detection.minBinSizeLum = minBinSizeLum;
    skin_detection.scale = (float)sigmaScale / 10.0;
    skin_detection.init(face.getFace(), face.getFaceMask());
    int key2 = cv::waitKey();

    if (key2 == 10)
    {
      // get color from skin
      //skin_points = face.getSkinPoints();
      //cout << "FOUND COLORS: " << skin_points[0] << endl;

      //skin_points.push_back(cv::Vec3b(178,114,146));
      //skin_points.push_back(cv::Vec3b(4,121,129));


      cout << "HOLD OBJECT TO CAMERA AND PRESS SPACE" << endl;
      cv::waitKey();

      mode = tracking;
    }

  }

  if (mode == tracking)
  {
    frame_counter++;

    // skip first 3 frames
    if (frame_counter < 3)
      return;

    DEBUG(2, cout << "Mode tracking" << endl);

    start = clock();
    //cv::Point3f nearest_point = getPOI(mycloud);

    cam.convertImage(*mycloud, image);
    ObjSegmentation seg(image, cam);
    skin_detection.scale = (float)sigmaScale/10.0;
    Mat skin_mask = skin_detection.getSkinMask(image);
    imshow("New Skin Mask", skin_mask);
    //seg.setSkinMask(skin_points, h_range, s_range, v_range, hyst_dist);
    seg.skin_mask = skin_mask;
    //cout << "Start Segmentation" << endl;



    pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    seg.useGrabCut = (bool)useGrabCut;
    seg.clusterObject(mycloud, obj_point);
    cv::Point3f new_obj_point = seg.getObjCenter();
    double obj_dist;

    // do not calc distance on first frame
    if (obj_point.x == 0 && obj_point.y == 0 && obj_point.z == 0)
      obj_dist = 0;
    else
      obj_dist = cv::norm(cv::Mat(obj_point),cv::Mat(new_obj_point));

    end = clock();
    DEBUG(0, cout << "Time required for segmentation: "<< (double)(end-start)/CLOCKS_PER_SEC << " seconds." << "\n\n");

    if (track_image)
    {
      if (obj_dist < 0.1)
      {
        obj_point = new_obj_point;
        trackImage(mycloud);
      }
      else
      {
        cout << "SKIP FRAME. OBjECT DISTANCE: " << obj_dist << endl;
        lost_pose_counter++;
      }
    }
    else
    {
      cam.convertImage(*mycloud, image);
      image.copyTo(im_draw);
      cv::imshow("image",im_draw);

    }


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


  // If user hits 'r' reset
  if (((char)key) == 'r')
  {
    mode = capture;
  }
  
  // If user hits 'r' reset
  if (((char)key) == 'p')
  {
    mode = parameters;
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
    cam.convertImage(*mycloud, image);
    imwrite( filename + ".jpg", image );
  }

  gend = clock();
  DEBUG(2, cout << "Time required for CB: "<< (double)(gend-gstart)/CLOCKS_PER_SEC << " seconds." << "\n\n");

}




/*********************************************************************
 * This function removes the background of the image
 ********************************************************************/
cv::Point3f getPOI(PointCloud<PointXYZRGB>::Ptr& cloud)
{
  float minz = 10000;
  int cnt_bad_point = 0;
  BOOST_FOREACH (pcl::PointXYZRGB& pt, cloud->points)
  {
    if (pt.z != pt.z || pt.x != pt.x || pt.y != pt.y)
    {
      //pt.r = 0;
      //pt.g = 0;
      //pt.b = 0;
      cnt_bad_point++;

    }
    /*	
        if(pt.z > float(max_camera_distance) / 100)
        {
        pt.x = bad_point;
        pt.y = bad_point;
        pt.z = bad_point;
        pt.r = 0;
        pt.g = 0;
        pt.b = 0;
        }*/

    if (pt.z < minz)
    {
      minz = pt.z;
      min_point.x = pt.x;
      min_point.y = pt.y;
      min_point.z = pt.z;
    }

    //printf ("\t(%f, %f, %f, %d, %d, %d)\n", pt.x, pt.y, pt.z, pt.r, pt.g, pt.b);

  }

  float error_value = (float)cnt_bad_point / cloud->points.size();
  view_error = (minz < (float)min_camera_distance / 100) || (error_value > 0.3);
}






/*********************************************************************
 * This function tracks one image
 ********************************************************************/
void trackImage(const PointCloud<PointXYZRGB>::ConstPtr& cloud)
{
  cout<<"---------------- FRAME #"<<i<<" -----------------------"<<endl;

  cam.convertImage(*cloud, image);
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



