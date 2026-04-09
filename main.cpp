// ===============================================
// Multiple Auto-Adapting Color Balancing Algorithm
// 多重自適應色彩平衡算法 - 完整實現
// 基於論文: "Multiple Auto-Adapting Color Balancing for Large Number of Images"
// ===========================AA==================

#include "Header.h"
#include <vector>
#include <iostream>
#include <stack>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <numeric>
#include <cstring>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;


// ===============================================
// 8. 主函數和使用範例
// ===============================================

// parameters
int height, width;
bool ignore_non_corrected, pano, ref_interval;
int main() {

    cout << "多重自適應色彩平衡算法" << endl;
    cout << "================================" << endl;


    string data_num_idx = "D:/DW_data/new_data/";
    vector<int> pairs;
    for (auto idx : fs::directory_iterator(data_num_idx)) {
        
        string data_num = idx.path().string();
        cout << "start correct: " << data_num << endl;
        vector<Mat> warped_imgs, masks;
        // 要用的data
        string warpDir = data_num + "/aligned_result/";
        string maskDir = data_num + "/img_masks/";
        string overlapDir = data_num + "/overlap/";
        string result_Dir = data_num + "/MACB_res/";
        warped_imgs = Inits::LoadImage(warpDir, IMREAD_COLOR);
        masks = Inits::LoadImgMask(maskDir);
        //讀overlap
        map<pair<int, int>, Mat> overlapmasks = Inits::LoadoverlapMask(overlapDir);
        height = warped_imgs[0].rows;
        width = warped_imgs[0].cols;
        pairs = Utils::MACB(warped_imgs, masks, result_Dir, overlapmasks);

        //// 創建排除遮罩 (可選)
        //Mat Adaptive_Excluding_Mask = createPercentageCutMask(inputImages[0]);

        //// 應用色彩平衡
        //cout << "\n開始色彩平衡處理..." << endl;
        //vector<Mat> balancedImages = balanceColors(inputImages, COLOR_GRID, Mat(), mask);

        //// 保存結果
        //cout << "\n保存結果..." << endl;
        //for (size_t i = 0; i < balancedImages.size(); i++) {
        //    if (!balancedImages[i].empty()) {
        //        string filename = "balanced_image_" + to_string(i) + ".jpg";
        //        bool success = imwrite(filename, balancedImages[i]);
        //        if (success) {
        //            cout << "已保存: " << filename << endl;
        //        } else {
        //            cout << "保存失敗: " << filename << endl;
        //        }

        //        // 同時保存原始圖像以便比較
        //        string originalFilename = "original_image_" + to_string(i) + ".jpg";
        //        imwrite(originalFilename, inputImages[i]);
        //    }
        //}

        //// 保存遮罩
        //if (!mask.empty()) {
        //    imwrite("exclusion_mask.jpg", mask * 255);
        //    cout << "已保存排除遮罩: exclusion_mask.jpg" << endl;
        //}

        //cout << "\n處理完成!" << endl;
        //cout << "請檢查輸出的圖像文件以查看色彩平衡效果。" << endl;

    }
    return 0;
}
