#include <opencv2/opencv.hpp>
#include <glob.h>
#include <fstream>

#define SHOW

void getImageFilenamesInVector(std::string dir, std::vector<std::string>& filenames)
{
    std::string glob_path{"../"+dir+"/"+dir+"/*.png"};
    glob_t result;
    glob(glob_path.c_str(), GLOB_TILDE, NULL,&result);
    for (size_t i{0}; i < result.gl_pathc; i++)
    {
        filenames.push_back(std::string(result.gl_pathv[i]));
    }
}

void getFeatures(const cv::Mat& gray_img, std::vector<cv::Point2f>& points)
{
    static int max_corners{500};
    static double quality{0.009};
    static double min_distance{9.0};

    cv::goodFeaturesToTrack(gray_img, points, max_corners, quality, min_distance);
}

void matchFeatures(const cv::Mat& img1, const cv::Mat& img2,
          std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2)
{
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(img1,img2,points1,points2,status,err);

    int idx{0};
    double pix_vel, min_vel_thresh{2.0}, max_vel_thresh{30.0};
    for (int i{0}; i < status.size(); i++)
    {
        cv::Point2f pt;
        pt = points2.at(i-idx);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0))
        {
            points1.erase(points1.begin() + (i-idx));
            points2.erase(points2.begin() + (i-idx));
            idx++;
        }
        else
        {
            double x_diff{points2[i-idx].x - points1[i-idx].x};
            double y_diff{points2[i-idx].y - points1[i-idx].y};
            pix_vel = sqrt(pow(x_diff,2) + pow(y_diff,2));

            if (pix_vel < min_vel_thresh || pix_vel > max_vel_thresh)
            {
                points1.erase(points1.begin() + (i-idx));
                points2.erase(points2.begin() + (i-idx));
                idx++;
            }
        }
    }
}

void runVO(std::string dir, int num_frames)
{
    cv::Mat M;
    cv::FileStorage fs("../"+dir+"/Camera_Parameters.yaml",cv::FileStorage::READ);
    fs["Cam_Mat"] >> M;
    fs.release();

    std::ofstream fout;
    fout.open("../"+dir+"/data.txt");

    std::vector<std::string> filenames;
    getImageFilenamesInVector(dir, filenames);

    cv::Mat prev_img, img, prev_gray, gray;
    std::vector<cv::Point2f> features, matches;

    cv::Mat R_tot, t_tot, R, t;
    R_tot = cv::Mat::eye(3,3,CV_64F);
    t_tot = cv::Mat::zeros(3,1,CV_64F);
    R = cv::Mat::eye(3,3,CV_64F);
    t = cv::Mat::zeros(3,1,CV_64F);
    double scale_factor{0.8};
    scale_factor *= (num_frames == 0) ? 1 : num_frames;

    int count{0};
    for (unsigned long i{0}; i < filenames.size(); i++)
    {
        prev_img = cv::imread(filenames[i]);
        if (prev_img.empty())
            break;
        cv::cvtColor(prev_img, prev_gray, cv::COLOR_BGR2GRAY);
        getFeatures(prev_gray, features);

        for (int j{0}; j < num_frames; j++)
            i++;

        img = cv::imread(filenames[i+1]);
        if (img.empty())
            break;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        matchFeatures(prev_gray,gray,features,matches);

        cv::Mat mask, E;
        if (matches.size() > 5)
        {
            E = cv::findEssentialMat(features,matches,M,cv::RANSAC,0.999,1.0,mask);
            cv::recoverPose(E, features, matches, M, R, t, mask);
        }
        else
        {
            count++;
            R = cv::Mat::eye(3,3,CV_64F);
            t = cv::Mat::zeros(3,1,CV_64F);
            t.at<double>(2,0) = 1.0;
        }

        R_tot *= R.t();
        t_tot += -scale_factor*R_tot*t;

        fout << R_tot.at<double>(0,0) << "\t" << R_tot.at<double>(0,1) << "\t";
        fout << R_tot.at<double>(0,2) << "\t" << t_tot.at<double>(0,0) << "\t";
        fout << R_tot.at<double>(1,0) << "\t" << R_tot.at<double>(1,1) << "\t";
        fout << R_tot.at<double>(1,2) << "\t" << t_tot.at<double>(1,0) << "\t";
        fout << R_tot.at<double>(2,0) << "\t" << R_tot.at<double>(2,1) << "\t";
        fout << R_tot.at<double>(2,2) << "\t" << t_tot.at<double>(2,0) << "\t\n";

#ifdef SHOW
        if (!E.empty())
        {
            std::vector<cv::Point2f> inliers1, inliers2;
            for (unsigned i{0}; i < matches.size(); i++)
            {
                cv::circle(img, features[i], 2, cv::Scalar{0,255,0},3);
                cv::circle(img, matches[i], 2, cv::Scalar{0,0,255},3);
                cv::line(img,features[i],matches[i],cv::Scalar{0,0,255},1);
            }
        }

        cv::imshow("Task 2",img);
        cv::waitKey(1);
#endif
    }
    std::cout << "Number of frames with features < 5: " << count << std::endl;
    fout.close();
#ifdef SHOW
    cv::destroyAllWindows();
#endif
}

int main()
{
    int skip_frames{2};

    std::string dir{"BYU_Hallway_Sequence"};
    runVO(dir, skip_frames);

    return 0;
}
