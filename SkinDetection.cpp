#include "SkinDetection.h"

SkinDetection::SkinDetection()
{
  minBinSize = 10;
  minBinSizeLum = 10;
  scale = 2;
}


bool SkinDetection::init(Mat face, Mat face_mask)
{
  Mat y_face;
  cvtColor(face, y_face, CV_BGR2YCrCb);
  int ymin;
  int ymax;

  this->calculateRangeHist(y_face, face_mask, ymin, ymax, minBinSizeLum, "Luminance Hist");


  cout << "YMIN: " << ymin << " | YMAX: " << ymax << endl;

  cv::Vec3b low(ymin, 0, 0);
  cv::Vec3b high(ymax, 255 , 255);
  
  // calculate new face mask with luminance range
  cv::inRange(y_face, low, high, face_mask);

  Mat myface;
  face.copyTo(myface, face_mask);
  imshow("Filtered Face", myface);

  Mat matNormR = Mat::zeros(face.rows, face.cols, CV_8UC1);
  Mat matNormG = Mat::zeros(face.rows, face.cols, CV_8UC1);
  Mat matR     = Mat::zeros(face.rows, face.cols, CV_8UC1); 

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

      if (rgbSum == 0)
        continue;
      
      float r = (float) (R / rgbSum);
      float g = (float) (G / rgbSum);

      matNormR.at<uint8_t>(j, i) = (uint8_t)(r * 255);
      matNormG.at<uint8_t>(j, i) = (uint8_t)(g * 255);
      matR.at<uint8_t>(j, i) = (uint8_t)R;
    }
  }

  int rmin=0, gmin=0, Rmin=0;
  int rmax=255, gmax=255, Rmax=255;
  // disable hist range
  this->calculateRangeHist(matNormR, face_mask, rmin, rmax, minBinSize, "Norm r histogram"); 
  this->calculateRangeHist(matNormG, face_mask, gmin, gmax, minBinSize, "Norm g histogram"); 
  this->calculateRangeHist(matR, face_mask, Rmin, Rmax, minBinSize, "R histogram"); 

  mu_r = 0;
  mu_g = 0;
  mu_R = 0;

  sigma_r = 0;
  sigma_g = 0;
  sigma_R = 0;

  int rcounter = 0;
  int gcounter = 0;
  int Rcounter = 0;

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

      if (rgbSum == 0)
        continue;

      float r = (float) (R / rgbSum);
      float g = (float) (G / rgbSum);


      // check borders
      if (rmin < r*255 && r*255 < rmax)
      {
        mu_r += r;
        rcounter++;
      }

      if (gmin < g*255 && g*255 < gmax)
      {
        mu_g += g;
        gcounter++;
      }

      if (Rmin < R && R < Rmax)
      {
        mu_R += R;
        Rcounter++;
      }
    }
  }

  mu_r = mu_r / rcounter;
  mu_g = mu_g / gcounter;
  mu_R = mu_R / Rcounter;


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
      if (rgbSum == 0)
        continue;
      float r = (float) (R / rgbSum);
      float g = (float) (G / rgbSum);

      if (rmin < r*255 && r*255 < rmax)
      {
        sigma_r += pow(r - mu_r, 2);
      }

      if (gmin < g*255 && g*255 < gmax)
      {
        sigma_g += pow(g - mu_g, 2);
      }

      if (Rmin < R && R < Rmax)
      {
        sigma_R += pow(R - mu_R, 2);
      }
    }
  }
  
  sigma_r = sqrt(sigma_r / rcounter);
  sigma_g = sqrt(sigma_g / gcounter);
  sigma_R = sqrt(sigma_R / Rcounter);


  _rLowBound = mu_r - scale * sigma_r;
  _gLowBound = mu_g - scale * sigma_g;
  _RLowBound = mu_R - scale * sigma_R * 2;

  _rUpBound = mu_r + scale * sigma_r;
  _gUpBound = mu_g + scale * sigma_g;
  _RUpBound = mu_R + scale * sigma_R;


  cout << "r-RANGE: " << _rLowBound << " | " << _rUpBound << endl;
  cout << "g-RANGE: " <<_gLowBound << " | " << _gUpBound << endl;
  cout << "R-RANGE: " << _RLowBound << " | " << _RUpBound << endl;

  return true;
}

void SkinDetection::calculateRangeHist(Mat img, Mat mask, int &min, int &max, int minBinEntries,  string imTitle)
{
  // Separate the image in 3 places ( B, G and R )
  vector<Mat> col_planes;
  split( img, col_planes );

  /// Establish the number of bins
  int histSize = 128;

  // Min number of values for a bin, not to be trimmed
  // minBinEntries = 10;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; 
  bool accumulate = false;

  Mat y_hist;

  /// Compute the histograms:
  calcHist( &col_planes[0], 1, 0, mask, y_hist, 1, &histSize, &histRange, uniform, accumulate );


  // delete peak at position 0
  y_hist.at<float>(0) = 0;

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
  
  //cout << "Max Loc: " << maxLoc << ": " << maxVal << endl;
    
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

  min = trim_left  * (256 / histSize);
  max = trim_right * (256 / histSize);
  /// Display
  imshow(imTitle, histImage );

}

Mat SkinDetection::getSkinMask(Mat image)
{
  _rLowBound = mu_r - scale * sigma_r;
  _gLowBound = mu_g - scale * sigma_g;
  _RLowBound = mu_R - scale * sigma_R * 2;

  _rUpBound = mu_r + scale * sigma_r;
  _gUpBound = mu_g + scale * sigma_g;
  _RUpBound = mu_R + scale * sigma_R;

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

  Mat tmp_mask = mask.clone();
  imshow("Skin mask before noise reduction", tmp_mask);

  // get rid of noise. 
  Mat kernel = Mat::ones(3,3, CV_8UC1);
  cv::erode(mask, mask, kernel);
  cv::dilate(mask, mask, kernel);
  cv::imshow("Noise reducing", mask);


  return mask;

}


