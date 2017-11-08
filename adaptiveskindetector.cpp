#include "adaptiveskindetector.hpp"

AdaptiveSkinDetector::AdaptiveSkinDetector()
{
    _hueLower=3;
    _hueUpper=33;

    //initialising the global skin color thresholds
    lower=cv::Scalar(3,50,50);
    higher=cv::Scalar(33,255,255);

    //the global histogram is given 0.95% weightage
    _mergeFactor=0.95;

    //setting the historgram computation parameters
    channels.resize(1);
    channels[0]=0;
    h.setChannel(channels);

    histSize.resize(1);
    histSize[0]=30;
    h.setHistSize(histSize);

    ranges.resize(2*channels.size());
    ranges[0]=0;
    ranges[1]=30;
    h.setRange(ranges);

}



void AdaptiveSkinDetector::run(cv::Mat image,cv::Mat &mask)
{

  cout << "TEST" << endl;


    cv::cvtColor(image,image,CV_BGR2HSV);
    //cv::inRange(image,lower,higher,mask);
    std::vector<cv::Mat> ch;
    cv::Mat hue;
    cv::split(image,ch);
    ch[0].copyTo(hue);

    //setting the mask for histogram
    h.setMask(mask);

   //build histogram based on global skin color threshols
    cv::Mat hist=h.BuildHistogram(hue,false);

    //normalize the histogram
    cv::normalize(hist,hist,0,1,cv::NORM_MINMAX);
    //update the histograms
    h.setHist(hist);

    //get the histogram thresholds
    std::vector<int> range1=h.getThreshHist(hist,0.05,0.05);

    _hueLower=range1[0];
    _hueUpper=range1[1];

    //obseve the pixels encountering motion
    cv::Mat mmask=cv::Mat();
    if(!p1.empty())
    {

      cv::Mat motion;
        cv::absdiff(p1,ch[2],motion);
        cv::inRange(motion,cv::Scalar(8,0,0),cv::Scalar(255,0,0),mmask);
        cv::erode(mmask,mmask,cv::Mat());
        cv::dilate(mmask,mmask,cv::Mat());
    }


    //compute a combined mask,representing motion of skin colored pixels
    if(!mmask.empty())
    cv::bitwise_and(mask,mmask,mmask);

    //set the new histogram mask
    h.setMask(mmask);

    //compute the histogram based on updated mask
    cv::Mat shist=h.BuildHistogram(hue,false);
    //normalize the histogram
    cv::normalize(shist,shist,0,1,cv::NORM_MINMAX);

    //merge both the histograms
    h.mergeHistogram(shist,0.02);

    //get the final histogram
    hist=h.getHist();

    //get the histogram thresholds
    h.getThreshHist(hist,0.05,0.05);

    //update the histogram thresholds
    _hueLower=range1[0];
    _hueUpper=range1[1];

    //Mat hist=h.BuildHistogram(hue,false);

    //comptute the new mask
    cv::MatIterator_<uchar> it = mask.begin<uchar>(), it_end = mask.end<uchar>();
    cv::MatIterator_<uchar> it1 = hue.begin<uchar>();
    for(; it != it_end; ++it,++it1++)
    {
        if(*it)
        {
            if(!(*it1>=_hueLower && *it1<=_hueUpper))
            {
                (*it)=0;
            }

        }

    }


    //store the current intensity image
    ch[2].copyTo(p1);
    cv::cvtColor(image,image,CV_HSV2BGR);



}
