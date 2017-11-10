#include "SkinDetection.h"

SkinDetection::SkinDetection()
{
}


bool SkinDetection::init(Mat face, Mat face_mask)
{
  Mat y_face;
  cvtColor(face, y_face, CV_BGR2YCrCb);
  
  this->calculateLuminanceRange(y_face, face_mask);


  cout << "YMIN: " << _ymin << " | YMAX: " << _ymax << endl;

  cv::Vec3b low(_ymin, 0, 0);
  cv::Vec3b high(_ymax, 255 , 255);
  
  // calculate new face mask with luminance range
  cv::inRange(y_face, low, high, face_mask);

  Mat myface;
  face.copyTo(myface, face_mask);
  imshow("Filtered Face", myface);


  //Mat norm_r = Mat::zeros(face.rows, face.cols, CV_8UC1);

  float mu_r = 0;
  float mu_g = 0;
  float mu_R = 0;

  float sigma_r = 0;
  float sigma_g = 0;
  float sigma_R = 0;

  int counter = 0;
  for (int i = 0; i < myface.cols; i++) 
  {
    for (int j = 0; j < myface.rows; j++) 
    {
      if (face_mask.at<char>(j, i) == 0)
        continue;

      counter++;
      Vec3b intensity = myface.at<Vec3b>(j, i);
      float B = intensity.val[0];
      float G = intensity.val[1];
      float R = intensity.val[2];

      float rgbSum = (float) (R + G + B);
      float r = (float) (R / rgbSum);
      float g = (float) (G / rgbSum);

      mu_r += r;
      mu_g += g;
      mu_R += R;

      //float blueNorm = (float) (B / rgbSum); 

      //res.at<Vec3b>(j, i)[0] = blueNorm;
      //res.at<Vec3b>(j, i)[1] = greenNorm;
      //res.at<Vec3b>(j, i)[2] = redNorm;
    }
  }

  mu_r = mu_r / counter;
  mu_g = mu_g / counter;
  mu_R = mu_R / counter;


  for (int i = 0; i < myface.cols; i++) 
  {
    for (int j = 0; j < myface.rows; j++) 
    {
      if (face_mask.at<char>(j, i) == 0)
        continue;

      Vec3b intensity = myface.at<Vec3b>(j, i);
      float B = intensity.val[0];
      float G = intensity.val[1];
      float R = intensity.val[2];

      float rgbSum = (float) (R + G + B);
      float r = (float) (R / rgbSum);
      float g = (float) (G / rgbSum);

      sigma_r += pow(r - mu_r, 2);
      sigma_g += pow(g - mu_g, 2);
      sigma_R += pow(R - mu_R, 2);
    }
  }

  sigma_r = sqrt(sigma_r / counter);
  sigma_g = sqrt(sigma_g / counter);
  sigma_R = sqrt(sigma_R / counter);

  
  _rLowBound = mu_r - 2 * sigma_r;
  _gLowBound = mu_g - 2 * sigma_g;
  _RLowBound = mu_R - 2 * sigma_R;

  _rUpBound = mu_r + 2 * sigma_r;
  _gUpBound = mu_g + 2 * sigma_g;
  _RUpBound = mu_R + 2 * sigma_R;


  cout << _rLowBound << " | " << _rUpBound << endl;
  cout << _gLowBound << " | " << _gUpBound << endl;
  cout << _RLowBound << " | " << _RUpBound << endl;

  return true;
}


void SkinDetection::calculateLuminanceRange(Mat face, Mat face_mask)
{
  // Separate the image in 3 places ( B, G and R )
  vector<Mat> col_planes;
  split( face, col_planes );

  /// Establish the number of bins
  int histSize = 128;

  // Min number of values for a bin, not to be trimmed
  int minBinEntries = 10;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; 
  bool accumulate = false;

  Mat y_hist, cr_hist, cb_hist;

  /// Compute the histograms:
  calcHist( &col_planes[0], 1, 0, face_mask, y_hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  //normalize(y_hist,  y_hist,  0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
    line( histImage, Point( bin_w*(i-1), hist_h - cvRound(y_hist.at<float>(i-1)) ) ,
        Point( bin_w*(i), hist_h - cvRound(y_hist.at<float>(i)) ),
        Scalar( 255, 0, 0), 2, 8, 0  );

  }

  // Trim the left side of the histogram by mirroring the right site
  double maxVal=0;
  Point maxLocPoint;
  minMaxLoc(y_hist, 0, &maxVal, 0, &maxLocPoint);

  int maxLoc = maxLocPoint.y;
  int trim_right = histSize - 1;

  for (int i = maxLoc; i < histSize; i++)
  {
    float val = y_hist.at<float>(i);
    if (val < minBinEntries)
    {
      trim_right = i;
      break;
    }
  }

  int trim_length = trim_right - maxLoc;
  int trim_left = maxLoc - trim_length;

  if (trim_left < 0)
    trim_left = 0;

  if (trim_right >= histSize)
    trim_right = histSize - 1;

  // draw left and right trim line
  line( histImage, Point( bin_w * trim_left, 0),
      Point( bin_w * trim_left, hist_h ),
      Scalar( 0, 255, 0), 2, 8, 0  );
  line( histImage, Point( bin_w * trim_right, 0),
      Point( bin_w * trim_right, hist_h ),
      Scalar( 0, 255, 0), 2, 8, 0  );

  _ymin = trim_left  * (256 / histSize);
  _ymax = trim_right * (256 / histSize);
  /// Display
  imshow("Luminance Histogram", histImage );

}

Mat SkinDetection::getSkinMask(Mat image)
{

  Mat mask  = Mat::zeros(image.rows, image.cols, CV_8UC1);
  
  for (int i = 0; i < image.cols; i++) 
  {
    for (int j = 0; j < image.rows; j++) 
    {

      Vec3b intensity = image.at<Vec3b>(j, i);
      float B = intensity.val[0];
      float G = intensity.val[1];
      float R = intensity.val[2];
      
      float rgbSum = (float) (R + G + B);
      float r = (float) (R / rgbSum);
      float g = (float) (G / rgbSum);

      if ( _rUpBound > r && r > _rLowBound &&
           _gUpBound > g && g > _gLowBound &&
           _RUpBound > R && R > _RLowBound)
      {
        mask.at<char>(j,i) = 255;
      }
    }
  }

  return mask;

}


