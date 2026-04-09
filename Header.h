#pragma once
#include <vector>
#include <iostream>
#include <stack>
#include <algorithm>
#include <fstream>
#include <filesystem>

#include <omp.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>    
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// 全局參數
const double default_percentage = 0.1;
const double constant_ratio = 2.844;
const double lower_cut_percentage = 0.075;
const double upper_cut_percentage = 0.005;
const int normalized_size = 256;

//===============================================
//數據結構定義
//===============================================

struct ADWParams {
    double windowSizePercentage;
    int numWindowsX;
    int numWindowsY;
    int windowWidth;
    int windowHeight;
    double overlapFactor;
};



struct polynomialcoeffs {
    vector<double> coeffs;
};


enum targetmodel {
    single_color = 0,
    color_grid = 1,
    first_order_poly = 2,
    second_order_poly = 3,
    third_order_poly = 4
};

namespace Inits
{
    vector<string> getImgFilenameList(string& addr);
    vector<Mat> LoadImage(string& addr, int flag);
    vector<Mat> LoadImgMask(string& addr);
    map<pair<int, int>, Mat> LoadoverlapMask(string& addr);
}

namespace Utils
{

    vector<pair<int, int>> getCorrectImgPair(vector<Mat> warped_img);
    //vector<Mat> balancecolors(const vector<Mat>& images, targetmodel model = color_grid,
    //    const Mat& externaltarget = Mat(), const Mat& mask = Mat());
    //vector<adaptivedodgingwindow> Utils::createadaptivedodgingwindows(const Mat& image, const Mat& mask_in, int overlapRatio = 0.5);
    ADWParams createadaptivedodgingwindows(const Mat& image, const Mat& mask_in);
    Mat createALMM(const cv::Mat& image, const Mat& mask, const ADWParams& params);
    Mat computeadaptivelocalmeanmap(const Mat& image, const Mat& mask);
    Mat bilinearInterpolateALMM(const Mat& almm, const ADWParams& params, const Mat& image, const Mat& mask);
    Mat single_img_correction(pair<int, int> tar_ref, vector<Mat>warped_imgs, vector<Mat> masks, map<pair<int, int>, Mat> overlapmasks);
    vector<int> MACB(vector<Mat> warped_imgs, vector<Mat> masks, string result_Dir, map<pair<int, int>, Mat> overlapmasks);
    Mat computesinglecolorsurface(const vector<Mat>& image, const Mat& externaltarget, const vector<Mat> mask, const Mat& externalmask);
    Mat computecolorgridsurface(const vector<Mat>& image, const vector<Mat>& mask, const Mat& gridsurface, int gridsize);
    Mat Polynomial_Surface(const vector<Mat>& image, const vector<Mat>& mask, const Mat& color_surface, int model, int gridsize);
    vector<double> fitPolynomial(const Mat& grid, int order, const Mat& image, int gridsize);
    Mat createPolyTargetMap(Size imageSize, const vector<double>& coeffs, int order, const vector<Mat>& mask);
    Mat computegridsurface(const vector<Mat>& image, const vector<Mat> mask, int gridsize);
    Mat balancecolors(const vector<Mat>& image, targetmodel model, const vector<Mat>& mask, int gridsize);
}
