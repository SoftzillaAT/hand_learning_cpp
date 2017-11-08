#ifndef ADAPTIVESKINDETECTOR_H
#define ADAPTIVESKINDETECTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "ColorHistogram.hpp"

/**
 * @brief The AdaptiveSkinColor class : Class incapsulating adaptive skin color detector
 * based on the paper An adaptive real-time skin detector based on Hue thresholding:
A comparison on two motion tracking methods by Farhad Dadgostar *, Abdolhossein Sarrafzadeh
 */


using namespace cv;
using namespace std;

class AdaptiveSkinDetector 
{
  cv::Mat image;

    //adaptive hue thresholds for skin color detection
    int _hueLower;
    int _hueUpper;

    //global lower and upper thresholds for skin color detection
    cv::Scalar lower;
    cv::Scalar higher;

    cv::Mat hist;

    //histogram merge factor for weighted average
    float _mergeFactor;

    //histogram paramters
    std::vector<int> histSize;
    std::vector<float> ranges;
    std::vector<int> channels;

    //object for histogram computation
    ColorHistogram h;

    //image required for image motion histogram
    cv::Mat p1;

public:
    /**
     * @brief AdaptiveSkinColor : constructor
     */
    AdaptiveSkinDetector();

    /**
     * @brief run : main function that performs adaptive skin color detection
     * @param image : input BGR image
     * @param mask : output mask ,1 are skin color pixels
     */
    void run(cv::Mat image, cv::Mat &mask);



};

#endif // ADAPTIVESKINDETECTOR_H
