#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <iostream>
#include <glob.h>
#include <vector>
#include <fstream>

#define SHOW

std::ofstream fout;

void getImageFiles(std::string glob_path, std::vector<std::string>& files);
void getFeatures(std::string first_img, cv::Mat& img, cv::Mat& gray, std::vector<cv::Point2d>& features);
void matchFeatures(cv::Mat& prev_gray, cv::Mat& gray, std::vector<cv::Point2d>& features, std::vector<cv::Point2d>& matches);
void acceptMatches(std::vector<cv::Point2d>& prev_features, std::vector<cv::Point2d>& features);
void calcTimeToImpact(std::vector<cv::Point2d>& prev_features, std::vector<cv::Point2d>& features);
cv::Point2d getPoint(cv::Point2d pt, int size, const cv::Mat& img);

int main()
{
    fout.open("../task1_data.txt");

    std::vector<std::string> img_files;
    std::string path{"../images/*.jpg"};
    getImageFiles(path, img_files);

    std::vector<cv::Point2d> features, matched_features, original_features;
    cv::Mat img_prev, img, gray_prev, gray;
    getFeatures(img_files[0], img_prev, gray_prev, original_features);

    for (unsigned long i{1}; i < img_files.size()-1; i++)
    {
        features = original_features;
        img = cv::imread(img_files[i]);
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        matchFeatures(gray_prev, gray, features, matched_features);
        acceptMatches(features, matched_features);

        calcTimeToImpact(features, matched_features);

        for (cv::Point2d pt : matched_features)
        {
            cv::Point point(int(pt.x), int(pt.y));
            cv::circle(img,point,2,cv::Scalar{0,255,0},-1);
        }

#ifdef SHOW
        cv::imshow("Matched Image", img);
        cv::waitKey(0);
#endif
    }

    fout.close();

    return 0;
}

void getImageFiles(std::string glob_path, std::vector<std::string>& files)
{
    glob_t result;
    glob(glob_path.c_str(), GLOB_TILDE, NULL,&result);
    for (size_t i{0}; i < result.gl_pathc; i++)
    {
        files.push_back(std::string(result.gl_pathv[i]));
    }
}

void getFeatures(std::string first_img, cv::Mat& img, cv::Mat& gray, std::vector<cv::Point2d>& features)
{
    img = cv::imread(first_img);
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    int x{285}, y{170}, w{80}, h{195};
    cv::Rect roi_can{x,y,w,h};

    int max_corners{100};
    double quality{0.01}, min_distance{10.0};
    cv::goodFeaturesToTrack(gray(roi_can),features, max_corners, quality, min_distance);

    for (unsigned long i{0}; i < features.size(); i++)
    {
        features[i].x += x;
        features[i].y += y;
        cv::Point point(int(features[i].x), int(features[i].y));
        cv::circle(img,point,2,cv::Scalar{0,0,255},-1);
    }

#ifdef SHOW
    cv::imshow("First Image", img);
    cv::waitKey(0);
#endif
}

void matchFeatures(cv::Mat& prev_gray, cv::Mat& gray, std::vector<cv::Point2d>& features, std::vector<cv::Point2d>& matches)
{
    matches.clear();
    int ts{5};
    int ss{12*ts};
    cv::Size template_size{ts,ts};
    cv::Size search_size{ss,ss};
    int method{cv::TM_SQDIFF_NORMED};
    for (cv::Point2d pt : features)
    {
        cv::Point2d template_pt{getPoint(pt,ts,prev_gray)};
        cv::Rect template_roi{template_pt,template_size};
        cv::Mat template_img{prev_gray(template_roi)};

        cv::Point2d search_pt{getPoint(pt,ss,prev_gray)};
        cv::Rect search_roi{search_pt,search_size};
        cv::Mat search_img{gray(search_roi)};

        int result_cols{search_img.cols - template_img.cols + 1};
        int result_rows{search_img.rows - template_img.rows + 1};
        cv::Mat result;
        result.create(result_rows, result_cols, CV_32FC1);
        cv::matchTemplate(search_img, template_img, result, method);
        cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

        double min_val, max_val;
        cv::Point match_loc, min_loc, max_loc;
        cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc, cv::Mat());
        match_loc.x = int(pt.x - result.cols/2.0 + min_loc.x);
        match_loc.y = int(pt.y - result_rows/2.0 + min_loc.y);

        matches.push_back(match_loc);
    }
}

void acceptMatches(std::vector<cv::Point2d>& prev_features, std::vector<cv::Point2d>& features)
{
    cv::Mat F, status;
    F = cv::findFundamentalMat(prev_features, features, cv::FM_RANSAC, 3, 0.99, status);

    std::vector<cv::Point2d> temp, prev_temp;
    for (int i{0}; i < status.rows; i++)
    {
        if (status.at<uchar>(i,0))
        {
            temp.push_back(features[i]);
            prev_temp.push_back(prev_features[i]);
        }
    }
    features = temp;
    prev_features = prev_temp;
}

void calcTimeToImpact(std::vector<cv::Point2d>& prev_features, std::vector<cv::Point2d>& features)
{
    static int count{1};
    double a{0}, num{0}, temp, tau{0};

    for (unsigned long i{0}; i < features.size(); i++)
    {
        a = (prev_features[i].x / features[i].x);
        temp = a / (a-1);

        if (temp > 0 && !std::isnan(temp) && !std::isinf(temp))
        {
            num++;
            tau += temp;
        }
    }

    tau /= num;

    fout << ++count << "\t" << tau << "\t\n";
}

cv::Point2d getPoint(cv::Point2d pt, int size, const cv::Mat& img)
{
    double x,y;
    if(pt.x > img.cols - size/2.0)
        x = img.cols - size;
    else if(pt.x < size/2.0)
        x = 0;
    else
        x = pt.x- size/2.0;
    if(pt.y > img.rows - size/2.0)
        y = img.rows - size;
    else if(pt.y < size/2.0)
        y = 0;
    else
        y = pt.y - size/2.0;

  return cv::Point2d(x, y);
}
