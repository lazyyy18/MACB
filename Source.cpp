#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <vector>
#include <iostream>
#include <stack>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <cMath>
#include <numeric>
#include <cstring>

#include "Header.h"




vector<string> Inits::getImgFilenameList(string& addr)
{
    vector<string> imgs;
    for (auto& entry : filesystem::directory_iterator(addr)) {
        if (entry.path().u8string().find(".png") != string::npos || entry.path().u8string().find(".JPG") != string::npos)
            imgs.push_back(entry.path().u8string());
    }
    cout << "getImgFilenameList finish" << endl;
    return imgs;
}

/**
 * 讀影像
 */
vector<Mat> Inits::LoadImage(string& addr, int flag)
{
    cout << "load warp images at " << addr  << "----->\n\n";
    vector<string> img_names = Inits::getImgFilenameList(addr);
    vector<Mat> imgs;
    imgs.reserve(img_names.size());
    for (const string& file_name : img_names)
        imgs.push_back(imread(file_name, flag));

    int N = (int)imgs.size();
    cout << "Num_imgs: " << N << endl;
    cout<< "load image finish \n\n";
    return imgs;
}

vector<Mat> Inits::LoadImgMask(string& addr)
{
    ofstream fout("output_gray.txt");
    vector<Mat> masks;
    Mat tmp_mask;
    vector<string> img_names = Inits::getImgFilenameList(addr);
    for (const string& file_name : img_names) {
        tmp_mask = imread(file_name, IMREAD_GRAYSCALE);
        // 寫入資料：每列對應一行像素
        masks.push_back(tmp_mask);
    }
    cout << masks.size() << endl;
    cout << "load mask finish \n\n";
    return masks;
}

//讀overlap
map<pair<int, int>, Mat> Inits::LoadoverlapMask(string& addr)
{
    vector<vector<int>> seq;
    //vector<vector<Mat>> compilation(N);
    map<pair<int, int>, Mat> masks;
    vector<string> img_names = Inits::getImgFilenameList(addr);
    cout << "load overlap images at " << addr << "----->\n\n";
    cout << img_names.size() << endl;
    for (const string& file_name : img_names) {
        cout << file_name << endl;
        size_t start = file_name.find_last_of("/") + 1;
        string filename_only = file_name.substr(start);

        int first_num = stoi(filename_only.substr(0, 2));
        int second_num = stoi(filename_only.substr(4, 2));
        cout << first_num << " , " << second_num << endl;
        //compilation[num].push_back(imread(file_name, IMREAD_GRAYSCALE));
        masks[make_pair(first_num, second_num)] = imread(file_name, IMREAD_GRAYSCALE);
    }
    cout << "load overlap_mask finish \n\n";
    cout << img_names.size();
    return masks;
}


vector<pair<int, int>> Utils::getCorrectImgPair(vector<Mat> warped_imgs) {
    vector<pair<int, int>> seq;
    seq.push_back(make_pair(0, 1));
    for (int i = 1; i < warped_imgs.size(); i++) {
        seq.push_back(make_pair(i, i - 1));
    }
    return seq;
}





// ===============================================
// 1. 基礎計算函數
// ===============================================

/**
 

// ===============================================
// 2. 自適應閃避窗口函數
// ===============================================

/**
 * 創建自適應閃避窗口
 */
ADWParams Utils::createadaptivedodgingwindows(const Mat& image, const Mat& mask_in)
{
    ADWParams params;
    // 計算全圖 mean 與 stddev
    Scalar meanVal, stddevVal;
    meanStdDev(image, meanVal, stddevVal, mask_in);
    double mu = meanVal[0];
    double sigma = stddevVal[0];
    if (sigma < 0.001) sigma = 0.001;
    cout << "mean_value" << meanVal << endl;
    cout << "std_value" << stddevVal << endl;
    // 計算 ρ（ADW 尺寸比例）
    double rho = (default_percentage / sigma) * (mu / constant_ratio);
    cout << "計算後 ADW 百分比 rho = " << rho << endl;
    params.windowSizePercentage = rho;
    double w = image.cols * rho;
    double h = image.rows * rho;
    // Calculate window dimensions
    params.windowWidth = static_cast<int>(15); // w
    params.windowHeight = static_cast<int>(15); // h
    cout << image.cols << ", " << image.rows << endl;
    // Ensure minimum window size
    params.windowWidth = max(5, params.windowWidth);
    params.windowHeight = max(5, params.windowHeight);
    cout << "ADW 尺寸（像素）= " << params.windowWidth << "///" << params.windowHeight << endl;
    // Calculate number of windows with overlap
    params.overlapFactor = 1.5;  // Increase number of windows for overlap
    params.numWindowsX = static_cast<int>(std::ceil(image.cols / (double)params.windowWidth * params.overlapFactor));
    params.numWindowsY = static_cast<int>(std::ceil(image.rows / (double)params.windowHeight * params.overlapFactor));
    cout << "切了x塊 " << params.numWindowsX << "///" << params.numWindowsY << endl;
    return params;

}
/**
 * 計算窗口的平均值
 */
Mat Utils::createALMM(const Mat& image, const Mat& mask, const ADWParams& params) {
    Mat almm(params.numWindowsY, params.numWindowsX, CV_64FC1);

    // Calculate step size for overlapping windows
    double stepX = (double)image.cols / params.numWindowsX;
    double stepY = (double)image.rows / params.numWindowsY;
    std::cout << "stepX: " << stepX << std::endl;
    std::cout << "stepY: " << stepY << std::endl;

    for (int n = 0; n < params.numWindowsY; n++) {
        for (int m = 0; m < params.numWindowsX; m++) {
            // Calculate window bounds
            int centerX = static_cast<int>((m + 0.5) * stepX);
            int centerY = static_cast<int>((n + 0.5) * stepY);

            int x1 = max(0, centerX - params.windowWidth / 2);
            int y1 = max(0, centerY - params.windowHeight / 2);
            int x2 = min(image.cols, x1 + params.windowWidth);
            int y2 = min(image.rows, y1 + params.windowHeight);

            // Extract window
            cv::Rect windowRect(x1, y1, x2 - x1, y2 - y1);
            cv::Mat windowImage = image(windowRect);
            cv::Mat windowMask = mask(windowRect);

            Scalar meanValues;
            if (windowImage.empty()) {
                meanValues = 0.0;
            }
            meanValues = mean(windowImage, windowMask);
            almm.at<double>(n, m) = meanValues[0];
        }
    }
    //cv::Mat almm_u8;
    //almm.convertTo(almm_u8, CV_8UC1);
    //cv::imshow("almm", almm_u8);
    return almm;
}


/// ===============================================
/// ALM插植法
/// ===============================================

Mat Utils::bilinearInterpolateALMM(const Mat& almm, const ADWParams& params, const Mat& image, const Mat& mask) {

    Size imageSize = image.size();
    cv::Mat result(imageSize, CV_64F, Scalar(0));

    double stepX = (double)imageSize.width / params.numWindowsX;
    double stepY = (double)imageSize.height / params.numWindowsY;

    for (int j = 0; j < imageSize.height; j++) {
        for (int i = 0; i < imageSize.width; i++) {
            if (mask.at<uchar>(j, i) == 0)
                continue;
            // Find which window this pixel belongs to
            double mx = i / stepX - 0.5;
            double my = j / stepY - 0.5;
            int m = static_cast<int>(std::floor(mx));
            int n = static_cast<int>(std::floor(my));

            // Ensure indices are within bounds
            m = std::max(0, std::min(m, params.numWindowsX - 2));
            n = std::max(0, std::min(n, params.numWindowsY - 2));


            double centerX = (m + 0.5) * stepX;
            double centerY = (n + 0.5) * stepY;
            int dx = (i >= centerX) ? 1 : -1;
            int dy = (j >= centerY) ? 1 : -1;
            if ((m + dx) < 0 || (m + dx) > almm.cols)
                dx = -dx;
            if ((n + dy) < 0 || (n + dy) > almm.rows)
                dy = -dy;
            Point2d v00_center = {centerX, centerY};
            Point2d v10_center = { (m + dx + 0.5) * stepX, (n + 0.5) * stepY };
            Point2d v01_center = { (m + 0.5) * stepX, (n + dy + 0.5) * stepY };
            Point2d v11_center = { (m + dx + 0.5) * stepX, (n + dy + 0.5) * stepY };
            // Calculate interpolation weights
            

            // Bilinear interpolation (Equation 4)
            double v00 = almm.at<double>(n, m);
            double v10 = almm.at<double>(n, m + dx);
            double v01 = almm.at<double>(n + dy, m);
            double v11 = almm.at<double>(n + dy, m + dx);

            double wx = std::abs(i - v00_center.x) / (v10_center.x - v00_center.x);
            double wy = std::abs(j - v00_center.y) / (v01_center.y - v00_center.y);

            double interpolatedValue = 
                (1 - wx) * (1 - wy) * v00 +
                wx * (1 - wy) * v10 +
                (1 - wx) * wy * v01 +
                wx * wy * v11;

            result.at<double>(j, i) = interpolatedValue;
        }
    }
    return result;
}

// ===============================================
// 3. 自適應局部平均圖計算 (方程式 3-4)
// ===============================================

/**
 * 計算自適應局部平均圖 (almm)
 */
Mat Utils::computeadaptivelocalmeanmap(const Mat& image, const Mat& mask) {

    cout << "計算自適應局部尺寸" << endl;
    //vector<adaptivedodgingwindow>  windows = Utils::createadaptivedodgingwindows(image, mask);
    ADWParams ADWP = Utils::createadaptivedodgingwindows(image, mask);
    cout << "---------finish--------" << endl;
    cout << "計算窗口平均值" << endl;
    Mat ALMN = Utils::createALMM(image, mask, ADWP);
    ///Utils::computewindowmeanvalues(image, ADWP, mask);
    cout << "---------finish--------" << endl;

   // Mat weightMap(image.size(), CV_32F, Scalar(0));
    cout << "計算局部平均圖" << endl;
    Mat localMeanMap = Utils::bilinearInterpolateALMM(ALMN, ADWP, image, mask);

    cv::Mat lmm_u8;
    localMeanMap.convertTo(lmm_u8, CV_8UC1);
    //cv::imshow("lmm", lmm_u8);
    //cv::waitKey(0);
    //cv::destroyAllWindows();

    cout << "---------finish--------" << endl;
    // 步驟5: 創建局部平均圖
    return localMeanMap;
}




// ===============================================
// 4. 自適應伽馬校正函數 (方程式 1-2)
// ===============================================

/**
 * 計算自適應伽馬值 (方程式 2)
 * γ(i,j) = log(t(i,j)) / log(m(i,j))
 */
double computeadaptivegamma(double targetvalue, double localmeanvalue) {
    targetvalue = max(0.001, min(1.0, targetvalue));
    localmeanvalue = max(0.001, min(1.0, localmeanvalue));

    double gamma = log(targetvalue) / log(localmeanvalue);
    return max(0.1, min(10.0, gamma));
}

/**
 * 應用伽馬校正 (方程式 1)
 * v_out(i,j) = α × v_in(i,j)^γ(i,j)
 */
double applygammacorrection(double inputvalue, double gamma, double alpha = 1.0) {
    inputvalue = max(0.001, min(1.0, inputvalue));
    double outputvalue = alpha * pow(inputvalue, gamma);
    return max(0.0, min(1.0, outputvalue));
}

// ===============================================
// 5. 目標色彩表面計算函數
// ===============================================

/**
 * 計算單一色彩目標表面 (方程式 6-8)
 */
Mat Utils::computesinglecolorsurface(const vector<Mat>& image, const Mat& externaltarget, const vector<Mat> mask, const Mat& externalmask) {
    cout << "計算單一色彩表面..." << endl;
    Size imageSize = image[0].size();
    Scalar globalmean, localmean;

    globalmean = mean(externaltarget, externalmask);
    localmean = mean(image, mask);

    cout << "全局平均值: " << globalmean[0]  << endl;
    cout << "局部平均值: " << localmean[0] << endl;

    Mat surface(imageSize, CV_64F, globalmean[0]);
    return surface;
}

Mat Utils::computegridsurface(const vector<Mat>& image, const vector<Mat> mask,int gridsize) {
    cout << "計算色彩網格表面..." << endl;
    Size imageSize = image[0].size();
    int m = ceil(image[0].cols / gridsize);
    int n = ceil(image[0].rows / gridsize);
    Mat gridsurface(n, m, CV_64FC1, Scalar(0));
    //Mat gridsurface = cv::Mat::zeros({ n, m }, CV_64FC1);
    Mat gridsurface_u8;
    // 使用第一個圖像計算網格 (簡化實現)
    cout << "色彩網格表面計算完成，網格數量: " << m << "x" << n << endl;
    for (int gy = 0; gy < n; gy++) {
        for (int gx = 0; gx < m; gx++) {
            int startx = gx * gridsize;
            int starty = gy * gridsize;
            int endx = min(startx + gridsize, image[0].cols);
            int endy = min(starty + gridsize, image[0].rows);
            Rect roi(startx, starty, endx - startx, endy - starty);
            double total_valid_pixels = 0;
            double gridtotalsum = 0;
            for (int image_idx = 0; image_idx < image.size(); image_idx++) {
                Mat sourceimage = image[image_idx];
                Mat sourcemask = mask[image_idx];

                // 若該區塊沒有被 mask 覆蓋（全為 0），則跳過
                if (countNonZero(sourcemask(roi)) == 0)
                    continue;
                int valid_count = countNonZero(sourcemask(roi));
                Scalar roi_mean = mean(sourceimage(roi), sourcemask(roi));
                gridtotalsum += roi_mean[0] * valid_count;
                total_valid_pixels += valid_count;
            }
            gridtotalsum /= total_valid_pixels;
            gridsurface.at<double>(gy, gx) = gridtotalsum;
        }
    }
    cout << image.size() << endl;
    gridsurface.convertTo(gridsurface_u8, CV_8UC1);
    //cv::imshow("gridsurface", gridsurface_u8);
    //cv::waitKey(0);
    //cv::destroyAllWindows();
    return gridsurface;

}


/**
 * 計算色彩網格目標表面 (方程式 9-11)
 */
Mat Utils::computecolorgridsurface(const vector<Mat>& image, const vector<Mat>& mask, const Mat& gridsurface, int gridsize) {
    Size imageSize = image[0].size();
    Mat result(imageSize, CV_64FC1, Scalar(0));
    for (int y = 0; y < image[0].rows; y++) {
        for (int x = 0; x < image[0].cols; x++) {
            bool haspixel = false;
            for (int image_idx = 0; image_idx < image.size(); image_idx++) {
                if (mask[image_idx].at<uchar>(y, x) != 0) {
                    haspixel = true;
                    break;
                }
            }
            if (haspixel == false)
                continue;
            // 找到對應 grid index
            double gx_f = (double)x / gridsize - 0.5;
            double gy_f = (double)y / gridsize - 0.5;

            int m = static_cast<int>(std::floor(gx_f));
            int n = static_cast<int>(std::floor(gy_f));

            // Ensure indices are within bounds
            m = std::max(0, std::min(m, gridsurface.cols - 2));
            n = std::max(0, std::min(n, gridsurface.rows - 2));

            double centerX = (m + 0.5) * gridsize;
            double centerY = (n + 0.5) * gridsize;


            int dx = (x >= centerX) ? 1 : -1;
            int dy = (y >= centerY) ? 1 : -1;
            if ((m + dx) < 0 || (m + dx) > gridsurface.cols)
                dx = -dx;
            if ((n + dy) < 0 || (n + dy) > gridsurface.rows)
                dy = -dy;
            cv::Point2d v00_center = { centerX, centerY };
            cv::Point2d v10_center = { (m + dx + 0.5 ) * gridsize, (n + 0.5) * gridsize };
            cv::Point2d v01_center = { (m + 0.5) * gridsize, (n + dy + 0.5) * gridsize };
            cv::Point2d v11_center = { (m + dx + 0.5) * gridsize, (n + dy + 0.5) * gridsize };
            // 找到網格索引

   
            // Bilinear interpolation (Equation 4)
            double v00 = gridsurface.at<double>(n, m);
            double v10 = gridsurface.at<double>(n, m + dx);
            double v01 = gridsurface.at<double>(n + dy, m);
            double v11 = gridsurface.at<double>(n + dy, m + dx);
            double wx = std::abs(x - v00_center.x) / (v10_center.x - v00_center.x);
            double wy = std::abs(y - v00_center.y) / (v01_center.y - v00_center.y);

            double interpolatedValue = (1 - wx) * (1 - wy) * v00 +
                wx * (1 - wy) * v10 +
                (1 - wx) * wy * v01 +
                wx * wy * v11;
            result.at<double>(y, x) = interpolatedValue;
        }
    }
    Mat result_U8;
    result.convertTo(result_U8, CV_8UC1);
    //cv::imshow("colorgridsurface", result_U8);
    //cv::waitKey(0);
    //cv::destroyAllWindows();
    return result;
}



Mat Utils::Polynomial_Surface(const vector<Mat>& image, const vector<Mat>& mask, const Mat& color_surface, int model, int gridsize) {
        cout << "計算 " << model << " 階多項式表面..." << endl;
        
        // 確保 color_surface 是 double 類型
        Mat grid;
        if (color_surface.type() != CV_64FC1) {
            color_surface.convertTo(grid, CV_64FC1);
        } else {
            grid = color_surface;
        }
        
        // 根據 model 選擇不同階數
        if (model == 1) {
            // 一階多項式 (線性平面)
            vector<double> coefficients = Utils::fitPolynomial(grid, 1, mask[0], gridsize);
            return createPolyTargetMap(image[0].size(), coefficients, 1, mask);
        } 
        else if (model == 2) {
            // 二階多項式 (二次曲面)
            vector<double> coefficients = Utils::fitPolynomial(grid, 2, mask[0], gridsize);
            return createPolyTargetMap(image[0].size(), coefficients, 2, mask);
        } 
        else {
            // 三階多項式 (三次曲面)
            vector<double> coefficients = Utils::fitPolynomial(grid, 3, mask[0], gridsize);
            return createPolyTargetMap(image[0].size(), coefficients, 3, mask);
        }
    }


vector<double> Utils::fitPolynomial(const Mat& grid, int order,const Mat& image, int gridsize) {
    // 收集網格點數據
    vector<Point2d> points;
    vector<double> values;
    int stepSize = gridsize;
    int imageWidth = image.cols ;
    int imageHeight = image.rows;
    for (int y = 0; y < grid.rows; y++) {
        for (int x = 0; x < grid.cols; x++) {
            double value = grid.at<double>(y, x);
            if (value > 0) {
                // 將 grid (gx, gy) 還原為 pixel 座標中心點
                int startX = x * stepSize;
                int startY = y * stepSize;
                int endX = std::min(startX + stepSize, imageWidth);
                int endY = std::min(startY + stepSize, imageHeight);

                int actualWidth = endX - startX;
                int actualHeight = endY - startY;

                double pixelX = startX + actualWidth / 2.0;
                double pixelY = startY + actualHeight / 2.0;

                double normX = pixelX / (imageWidth - 1);
                double normY = pixelY / (imageHeight - 1);

                points.emplace_back(normX, normY);
                values.push_back(value);
            }
        }
    }

    // 確定係數數量
    int numCoeffs;
    if (order == 1) numCoeffs = 3;
    else if (order == 2) numCoeffs = 6;
    else numCoeffs = 10;

    // 建立設計矩陣
    int n = points.size();
    Mat X(n, numCoeffs, CV_64FC1);
    Mat Y(n, 1, CV_64FC1);

    for (int i = 0; i < n; i++) {
        double x = points[i].x;
        double y = points[i].y;

        if (order == 1) {
            // B = [1, x, y]
            X.at<double>(i, 0) = 1.0;
            X.at<double>(i, 1) = x;
            X.at<double>(i, 2) = y;
        }
        else if (order == 2) {
            // B = [1, x, y, x², x·y, y²]
            X.at<double>(i, 0) = 1.0;
            X.at<double>(i, 1) = x;
            X.at<double>(i, 2) = y;
            X.at<double>(i, 3) = x * x;
            X.at<double>(i, 4) = x * y;
            X.at<double>(i, 5) = y * y;
        }
        else {  // order == 3
            // B = [1, x, y, x², x·y, y², x³, x²·y, x·y², y³]
            X.at<double>(i, 0) = 1.0;
            X.at<double>(i, 1) = x;
            X.at<double>(i, 2) = y;
            X.at<double>(i, 3) = x * x;
            X.at<double>(i, 4) = x * y;
            X.at<double>(i, 5) = y * y;
            X.at<double>(i, 6) = x * x * x;
            X.at<double>(i, 7) = x * x * y;
            X.at<double>(i, 8) = x * y * y;
            X.at<double>(i, 9) = y * y * y;
        }

        Y.at<double>(i, 0) = values[i];
    }

    // 解最小平方問題
    Mat coeffMat;
    solve(X, Y, coeffMat, DECOMP_SVD);

    // 轉換為 vector
    vector<double> coefficients(numCoeffs);
    for (int i = 0; i < numCoeffs; i++) {
        coefficients[i] = coeffMat.at<double>(i, 0);
    }

    // 輸出係數
    cout << "多項式係數: ";
    for (double c : coefficients) {
        cout << c << " ";
    }
    cout << endl;

    return coefficients;
}

Mat Utils::createPolyTargetMap(Size imageSize, const vector<double>& coeffs, int order, const vector<Mat>& mask) {
    Mat targetMap(imageSize, CV_64FC1,Scalar(0));
    for (int j = 0; j < imageSize.height; j++) {
        for (int i = 0; i < imageSize.width; i++) {
            bool haspixel = false;
            for (int image_idx = 0; image_idx < mask.size(); image_idx++) {
                if (mask[image_idx].at<uchar>(j, i) != 0) {
                    haspixel = true;
                    break;
                }
            }
            if (haspixel == false)
                continue;

            // 正規化座標
            double x = (double)i / (imageSize.width - 1);
            double y = (double)j / (imageSize.height - 1);

            double value = 0.0;

            if (order == 1) {
                // T(i,j) = a0 + a1*x + a2*y
                value = coeffs[0] + coeffs[1] * x + coeffs[2] * y;
            }
            else if (order == 2) {
                // T(i,j) = a0 + a1*x + a2*y + a3*x² + a4*x*y + a5*y²
                value = coeffs[0] + coeffs[1] * x + coeffs[2] * y +
                    coeffs[3] * x * x + coeffs[4] * x * y + coeffs[5] * y * y;
            }
            else {  // order == 3
                // T(i,j) = a0 + a1*x + ... + a9*y³
                value = coeffs[0] + coeffs[1] * x + coeffs[2] * y +
                    coeffs[3] * x * x + coeffs[4] * x * y + coeffs[5] * y * y +
                    coeffs[6] * x * x * x + coeffs[7] * x * x * y +
                    coeffs[8] * x * y * y + coeffs[9] * y * y * y;
            }

            targetMap.at<double>(j, i) = value;
        }
    }

    return targetMap;
}

// ===============================================
// 7. 主要處理函數
// ===============================================

/**
 * 處理單個圖像
 */
Mat processimage(const Mat& image, const Mat& targetsurface, targetmodel model, const Mat& mask = Mat()) {
    cout << "處理圖像，大小: " << image.cols << "x" << image.rows << endl;

}

/**
 * 主要色彩平衡函數
 */
Mat Utils::balancecolors(const vector<Mat>& image, targetmodel model, const vector<Mat>& mask , int gridsize) {

    // 計算目標色彩表面
    Mat targetsurface,gridsurface,gridsurface_u8;;
    switch (model) {
    //case single_color:
    //    targetsurface = Utils::computesinglecolorsurface(image, externaltarget, mask, externakmask);
    //    break;
    case color_grid:
        gridsurface  = Utils::computegridsurface(image, mask, gridsize);
        targetsurface = Utils::computecolorgridsurface(image, mask, gridsurface, gridsize);
        break;
    case first_order_poly:
        gridsurface = Utils::computegridsurface(image, mask, gridsize);
        targetsurface = Utils::Polynomial_Surface(image, mask, gridsurface, 1, gridsize);
        break;

    case second_order_poly:
        gridsurface = Utils::computegridsurface(image, mask, gridsize);
        targetsurface = Utils::Polynomial_Surface(image, mask, gridsurface, 2, gridsize);
        break;

    case third_order_poly:
        gridsurface = Utils::computegridsurface(image, mask, gridsize);
        targetsurface = Utils::Polynomial_Surface(image, mask, gridsurface, 3, gridsize);
        break;
    }
    //cv::Mat targetsurface_u8;
    //targetsurface.convertTo(targetsurface_u8, CV_8UC1);
    //cv::imshow("targetsurface", targetsurface_u8);
    return targetsurface;
}

Mat Utils::single_img_correction(pair<int, int> tar_ref, vector<Mat>warped_imgs, vector<Mat> masks, map<pair<int, int>, Mat> overlapmasks)
{   
    cout << "圖片長: " << warped_imgs[0].rows << "圖片寬: " << warped_imgs[0].cols << endl;
    int choice = 1;
    int gridsize = 15;
    cout << "請選擇 target model：" << endl;
    cout << " 0 - single_color" << endl;
    cout << " 1 - color_grid" << endl;
    cout << " 2 - first_order_poly" << endl;
    cout << " 3 - second_order_poly" << endl;
    cout << " 4 - third_order_poly" << endl;
    cout << "你的選擇：";
    //cin >> choice;
    targetmodel model = static_cast<targetmodel>(choice);
    int tar = tar_ref.first;
    int ref = tar_ref.second;
    double alpha = 1.0;

    vector<Mat> corrected_imgs_hsv(3), mergeT(3), mergeW(3);
    vector<Mat> wrap_channels(warped_imgs.size());
    Mat wraped_tar_imgs_ycrcb;
    vector<Mat> imgschannels(3);

    for (int c = 0; c < 3; c++) {
        for (int image_idx = 0; image_idx < warped_imgs.size(); image_idx++) {
            cvtColor(warped_imgs[image_idx], wraped_tar_imgs_ycrcb, COLOR_BGR2YCrCb);
            split(wraped_tar_imgs_ycrcb, imgschannels);
            wrap_channels[image_idx] = imgschannels[c].clone();  // deep copy 避免錯誤
        }

        // 目標圖像準備
        Mat correctedImage(wrap_channels[tar].rows, wrap_channels[tar].cols, CV_64FC1, Scalar(0));
        Mat input_tar_normalized;
        wrap_channels[tar].convertTo(input_tar_normalized, CV_64FC1, 1.0 / 255.0); // 輸入應為 64F

        Mat W = Utils::computeadaptivelocalmeanmap(wrap_channels[tar], masks[tar]);
        Mat T = Utils::balancecolors(wrap_channels, model, masks, gridsize);
        mergeT[c] = T;
        mergeW[c] = W;
        Mat gamma_map(wrap_channels[tar].rows, wrap_channels[tar].cols, CV_64FC1, Scalar(1.0));

        for (int y = 0; y < W.rows; ++y) {
            for (int x = 0; x < W.cols; ++x) {
                if (masks[tar].at<uchar>(y, x) == 0) continue;

                double w = W.at<double>(y, x);
                double t = T.at<double>(y, x);

                if (w > 1e-6 && t > 1e-6)
                    gamma_map.at<double>(y, x) = t / w;//std::log(t) / std::log(w);
                //cout << std::log(t) / std::log(w) << endl;
                double v_in = input_tar_normalized.at<double>(y, x);
                double v_out = v_in * gamma_map.at<double>(y, x);
                correctedImage.at<double>(y, x) = std::min(std::max(v_out, 0.0), 1.0);
            }
        }

        correctedImage.convertTo(corrected_imgs_hsv[c], CV_8UC1, 255.0);
    }

    // 合併輸出影像與 T 圖
    Mat result_hsv, result_tmp, result, tmp_T, tmp_W, tmp_T_, tmp_W_;
    merge(corrected_imgs_hsv, result_hsv);
    merge(mergeT, tmp_T);
    merge(mergeW, tmp_W);
    tmp_T.convertTo(tmp_T_, CV_8UC3);
    tmp_W.convertTo(tmp_W_, CV_8UC3);
    cvtColor(tmp_T_, tmp_T, COLOR_YCrCb2BGR);
    cvtColor(tmp_W_, tmp_W, COLOR_YCrCb2BGR);
    cvtColor(result_hsv, result_tmp, COLOR_YCrCb2BGR);

    result = cv::Mat::zeros(result_tmp.size(), CV_8UC3);
    result_tmp.copyTo(result, masks[tar]);
    //imshow("W", tmp_W);
    //imshow("T", tmp_T);
    //imshow("result", result);
    waitKey(0);

    return result;

}

vector<int> Utils::MACB(vector<Mat> warped_imgs, vector<Mat> masks, string result_Dir, map<pair<int, int>, Mat> overlapmasks)
{
    string output_dir = result_Dir;
    if (!filesystem::exists(output_dir)) {
        filesystem::create_directories(output_dir); // C++17
    }
    //panorama = Mat::zeros(warped_imgs[0].size(), CV_8UC3);
    //get the first image to correct
    vector<bool> is_corrected(warped_imgs.size(), false);
    //int i = Utils::getNextImageID(is_corrected);

    vector<pair<int, int>> correct_seq = Utils::getCorrectImgPair(warped_imgs);

    vector<int> final_sequence;
    //删除v第一个元素
    //vector<pair<int, int>>::iterator k = correct_seq.begin();
    //correct_seq.erase(k);
    //const pair<int, int>& tar_ref = correct_seq[0];
    //imwrite(output_dir + "00__warped_img.png", warped_imgs[0]);
    for (const pair<int, int>& tar_ref : correct_seq) {
        cout << "\n******* Correcting image " << "tar: " << tar_ref.first << "  ref: " << tar_ref.second << " *******\n";
        Mat corrected_images = single_img_correction(tar_ref, warped_imgs, masks, overlapmasks);
        imwrite(output_dir + "0" + to_string(tar_ref.first) + "__warped_img.png", corrected_images);
        final_sequence.push_back(tar_ref.first);
        is_corrected[tar_ref.first] = true;
    }

    cout << "\nCorrection sequence: ";
    for (int& seq : final_sequence) cout << seq << " ";
    cout << endl;

    return final_sequence;
}

