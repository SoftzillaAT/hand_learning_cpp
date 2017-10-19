#ifndef PCLMANIPULATION_H
#define PCLMANIPULATION_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

using namespace pcl;
using namespace cv;
using namespace std;

class PclManipulation
{
	private:
		static bool lexico_compare2d(const cv::Point2d&, const cv::Point2d&);
		static bool points_are_equal2d(const cv::Point2d&, const cv::Point2d&);

	public:
		static std::vector<cv::Point3f> calcMeanShiftPoints(PointCloud<PointXYZRGB>::Ptr& cloud, int num_particles, float mean_shift_radius);
		static void calcMeanShift(cv::Point3f &p, PointCloud<PointXYZRGB>::Ptr& cloud, float radius);

		static pcl::PointCloud<PointXYZRGB>::Ptr createCloud(std::vector<cv::Point3f> points);
		static cv::Point3f getClusterPoint(std::vector<cv::Point3f> points, float cluster_dist);

		static void clusterCloud(pcl::PointCloud<PointXYZRGB>::Ptr& cloud, cv::Point2d p);

		static std::vector<int> getNearestNeigboursInRadius(pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree, float radius, cv::Point3f p);
		

		
static void clusterCloud2(PointCloud<PointXYZRGB>::Ptr& cloud, PointCloud<PointXYZRGB>::Ptr& cloud_result);


};

#endif
