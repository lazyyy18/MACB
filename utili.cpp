#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <typeinfo>
#include <fstream>
#include <numeric>
#include <cstring>

#include "utili.h"

vector<string> Inits::getImgFilenameList(string& addr)
{
    vector<string> imgs;
    for (auto& entry : filesystem::directory_iterator(addr)) {
        if (entry.path().u8string().find(".png") != string::npos || entry.path().u8string().find(".JPG") != string::npos)
            imgs.push_back(entry.path().u8string());
    }
    return imgs;
}

/**
 * Åª¼v¹³
 */
vector<Mat> Inits::LoadImage(string& addr, int flag)
{
    vector<string> img_names = Inits::getImgFilenameList(addr);
    vector<Mat> imgs;
    imgs.reserve(img_names.size());
    for (const string& file_name : img_names)
        imgs.push_back(imread(file_name, flag));

    return imgs;
}

vector<Mat> Inits::LoadImgMask(string& addr, int N)
{
    vector<Mat> masks;
    vector<string> img_names = Inits::getImgFilenameList(addr);

    for (const string& file_name : img_names) {
        masks.push_back(imread(file_name, IMREAD_GRAYSCALE));
    }

    cout << masks.size();
    return masks;
}

/**
 * ÅªÀÉ
 */
void Inits::loadAll(string& warpDir, string& maskDir)
{
    cout << "load warp images-------------------->\n\n";
    warped_imgs = Inits::LoadImage(warpDir, IMREAD_COLOR);
    int N = (int)warped_imgs.size();
    cout << N << " images loaded.\n\n";
    cout << "load Mask-------------------->\n\n";
    masks = Inits::LoadImgMask(maskDir, N);
    cout << " mask loaded.\n\n";
}

vector<pair<int, int>> Utils::getCorrectImgPair() {
    vector<pair<int, int>> seq;
    seq.push_back(make_pair(0, 1));
    for (int i = 1; i < warped_imgs.size(); i++) {
        seq.push_back(make_pair(i, i - 1));
    }
    return seq;
}

vector<int> Utils::MACB()
{
    //panorama = Mat::zeros(warped_imgs[0].size(), CV_8UC3);
    //get the first image to correct
    vector<bool> is_corrected(warped_imgs.size(), false);
    //int i = Utils::getNextImageID(is_corrected);

    vector<pair<int, int>> correct_seq = Utils::getCorrectImgPair();

    vector<int> final_sequence;
    for (const pair<int, int>& tar_ref : correct_seq) {
        cout << "\n******* Correcting image " << tar_ref.first << " *******\n";
        final_sequence.push_back(tar_ref.first);
        is_corrected[tar_ref.first] = true;
    }

    cout << "\nCorrection sequence: ";
    for (int& seq : final_sequence) cout << seq << " ";
    cout << endl;

    return final_sequence;
}

