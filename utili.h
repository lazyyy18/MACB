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

extern vector<Mat> warped_imgs, masks;


using namespace std;
using namespace cv;

namespace Inits
{
    vector<string> getImgFilenameList(string& addr);
    vector<Mat> LoadImage(string& addr, int flag);
    vector<Mat> LoadImgMask(string& addr, int N);
    void loadAll(string& warpDir, string& maskDir);
}

namespace Utils
{
    vector<pair<int, int>> getCorrectImgPair();  
    vector<int> MACB();
}
