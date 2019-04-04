#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <iostream>
#include <glob.h>
#include <vector>

#define SHOW

int mouseX, mouseY;

void firstCallback(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        mouseX = x;
        mouseY = y;
        std::cout << "x: " << x << std::endl << "y: " << y << std::endl;
    }
}

void lastCallback(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        mouseX = x;
        mouseY = y;
        std::cout << "x: " << x << std::endl << "y: " << y << std::endl;
    }
}

double mag(double in)
{
    in *= (in < 0) ? -1 : 1;
    return in;
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

void featureMatch(const std::vector<cv::Point2d>& prev_corners, const cv::Mat& prev_img,
                  const cv::Mat& img, std::vector<cv::Point2d>& new_corners)
{
    new_corners.clear();
    int ts{5};
    int ss{12*ts};
    cv::Size template_size{ts,ts};
    cv::Size search_size{ss,ss};
    int method{cv::TM_SQDIFF_NORMED};
    for (cv::Point2d pt : prev_corners)
    {
        cv::Point2d template_pt{getPoint(pt,ts,prev_img)};
        cv::Rect template_roi{template_pt,template_size};
        cv::Mat template_img{prev_img(template_roi)};

        cv::Point2d search_pt{getPoint(pt,ss,prev_img)};
        cv::Rect search_roi{search_pt,search_size};
        cv::Mat search_img{img(search_roi)};

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

        new_corners.push_back(match_loc);
    }
}

void getF(std::string glob_path, std::vector<cv::Point2d>& original_pts,
             std::vector<cv::Point2d>& end_pts, cv::Mat& F, cv::Mat& first_img,
          cv::Mat& last_img)
{
    // get all of the img filenames in a vector
    glob_t result;
    glob(glob_path.c_str(), GLOB_TILDE, NULL,&result);
    std::vector<std::string> filenames;
    for (size_t i{0}; i < result.gl_pathc; i++)
    {
        filenames.push_back(std::string(result.gl_pathv[i]));
    }
    // grab the first image
    cv::Mat prev_img, prev_img_g, img, img_g;
    prev_img = cv::imread(filenames[0]);
    cv::cvtColor(prev_img,prev_img_g,cv::COLOR_BGR2GRAY);
    prev_img.copyTo(first_img);

    std::vector<cv::Point2d> prev_corners, original_corners, new_corners;
    int max_corners{500};
    double quality{0.01}, min_distance{25.0};
    cv::goodFeaturesToTrack(prev_img_g, prev_corners, max_corners, quality, min_distance);
    original_corners = prev_corners;

    for (std::string filename : filenames)
    {
        img = cv::imread(filename);
        cv::cvtColor(img, img_g, cv::COLOR_BGR2GRAY);
        cv::Mat temp_img;
        img.copyTo(temp_img);

        featureMatch(prev_corners, prev_img_g, img_g, new_corners);

        cv::FileStorage in{"../Camera_Parameters.yaml", cv::FileStorage::READ};
        cv::Mat M, dist;
        in["Cam_Mat"] >> M;
        in["Distortion"] >> dist;
        in.release();
        double fx{M.at<double>(0,0)};
        double fy{M.at<double>(1,1)};
        double Ox{M.at<double>(0,2)};
        double Oy{M.at<double>(1,2)};

        cv::Mat status;
        cv::undistortPoints(prev_corners, prev_corners, M, dist);
        cv::undistortPoints(new_corners, new_corners, M, dist);
        for (int i{0}; i < prev_corners.size(); i++)
        {
            prev_corners[i].x = prev_corners[i].x*fx + Ox;
            prev_corners[i].y = prev_corners[i].y*fy + Oy;
            new_corners[i].x = new_corners[i].x*fx + Ox;
            new_corners[i].y = new_corners[i].y*fy + Oy;
        }
        F = cv::findFundamentalMat(prev_corners,new_corners,cv::FM_RANSAC,3,0.99,status);

        prev_corners.clear();
        std::vector<cv::Point2d> temp;
        for (int i{0}; i < status.rows; i++)
        {
            if (status.at<uchar>(i,0))
            {
                prev_corners.push_back(new_corners[i]);
                temp.push_back(original_corners[i]);
            }
        }
        original_corners = temp;
        img_g.copyTo(prev_img_g);

        for (int i{0}; i < prev_corners.size(); i++)
        {
            cv::circle(temp_img, original_corners[i], 2, cv::Scalar(0,255,0), -1);
            cv::line(temp_img, original_corners[i], prev_corners[i], cv::Scalar(0,0,255),1);
        }
#ifdef SHOW
        cv::imshow("Debug",temp_img);
        cv::waitKey(0);
#endif
    }
    original_pts = original_corners;
    end_pts = prev_corners;
    cv::Mat status;
    F = cv::findFundamentalMat(original_pts,end_pts,cv::FM_RANSAC,3,0.99,status);
    last_img = cv::imread(filenames[5]);
}

void rectify(std::string dir)
{
    std::string path{"../"+dir+"/*.jpg"};
    std::vector<cv::Point2d> original_pts, final_pts;
    cv::Mat F, first_img, last_img;
    getF(path, original_pts, final_pts, F, first_img, last_img);

    cv::FileStorage in{"../Camera_Parameters.yaml", cv::FileStorage::READ};
    cv::Mat M, dist;
    in["Cam_Mat"] >> M;
    in["Distortion"] >> dist;
    in.release();

    cv::Mat E, R1, R2, t, R;
    E = M.t() * F * M;
    cv::decomposeEssentialMat(E,R1, R2, t);

    double e1{3 - mag(R1.at<double>(0,0)) - mag(R1.at<double>(1,1)) - mag(R1.at<double>(2,2))};
    double e2{3 - mag(R2.at<double>(0,0)) - mag(R2.at<double>(1,1)) - mag(R2.at<double>(2,2))};

    std::cout << "Current image set: " << dir << std::endl;
    if (dir.substr(0,8) == "Parallel")
    {
        if (e1 < e2)
        {
            R = R1;
            std::cout << "R" << R1 << std::endl;
        }
        else
        {
            R = R2;
            std::cout << "R" << R2 << std::endl;
        }
        t = t * 2.7;
    }
    else
    {
        if (R1.at<double>(1,1) > 0)
        {
            R = R1;
            std::cout << "R" << R1 << std::endl;
        }
        else
        {
            R = R2;
            std::cout << "R" << R2 << std::endl;
        }
        t = t * 2.45;
    }
    if (t.at<double>(0) > 0)
        std::cout << "t" << t << std::endl;
    else
    {
        t = -t;
        std::cout << "t" << -t << std::endl;
    }

    cv::Size img_size{640,480};
    cv::Mat P1, P2, Q;
    cv::stereoRectify(M,dist,M,dist,img_size, R, t, R1, R2, P1, P2, Q);

    std::vector<cv::Point3d> last_points, first_points;
    for (int i{0}; i < 4; i++)
    {
        int s{1};
        int a{0};
        double fx, fy, ox, oy;
        fx = final_pts[s*i+a].x;
        fy = final_pts[s*i+a].y;
        last_points.push_back(cv::Point3d{fx,fy,0.0});
        cv::circle(last_img, cv::Point{int(fx),int(fy)},5,cv::Scalar(0,0,255),-1);
        ox = original_pts[s*i+a].x;
        oy = original_pts[s*i+a].y;
        first_points.push_back(cv::Point3d{ox,oy,fx-ox});
        cv::circle(first_img, cv::Point{int(ox),int(oy)},5,cv::Scalar(0,0,255),-1);
    }

    std::vector<cv::Point3d> obj_points;
    cv::perspectiveTransform(first_points, obj_points, Q);

    std::cout << "3D point estimates:" << std::endl;
    for (cv::Point3d obj_pt : obj_points)
        std::cout << obj_pt << std::endl;

    cv::imshow("Original 4 Points", first_img);
    cv::imshow("Final 4 Points", last_img);
    cv::waitKey(0);

    // parallel points on closest edge
//    cv::Point3d pt4_R{408.0, 423.0, 0.0};
//    cv::Point3d pt3_R{408.0, 313.0, 0.0};
//    cv::Point3d pt2_R{407.0, 201.0, 0.0};
//    cv::Point3d pt1_R{405.0, 92.0, 0.0};

//    cv::Point3d pt4_L{297.5, 424.0, -297.5+pt4_R.x};
//    cv::Point3d pt3_L{297.0, 313.0, -297.0+pt3_R.x};
//    cv::Point3d pt2_L{296.0, 202.0, -296.0+pt2_R.x};
//    cv::Point3d pt1_L{295.0, 92.0, -295.0+pt1_R.x};
    // turned points on closest edge
//    cv::Point3d pt4_R{336.0, 408.0, 0.0};
//    cv::Point3d pt3_R{336.0, 306.0, 0.0};
//    cv::Point3d pt2_R{334.0, 202.0, 0.0};
//    cv::Point3d pt1_R{333.0, 99.0, 0.0};

//    cv::Point3d pt4_L{235.0, 414.0, -235.0+pt4_R.x};
//    cv::Point3d pt3_L{234.0, 306.0, -234.0+pt3_R.x};
//    cv::Point3d pt2_L{234.0, 200.0, -234.0+pt2_R.x};
//    cv::Point3d pt1_L{233.0, 94.0, -233.0+pt1_R.x};

//    std::vector<cv::Point3d> points{pt1_L, pt2_L, pt3_L, pt4_L};

//    std::vector<cv::Point3d> obj_points;
//    cv::perspectiveTransform(points, obj_points, Q);

//    for (int i{0}; i < obj_points.size(); i++)
//    {
//        std::cout << i << std::endl << obj_points[i] << std::endl;
//    }

//    cv::imshow("first", first_img);
//    cv::imshow("last", last_img);
//    cv::setMouseCallback("first", firstCallback);
//    cv::setMouseCallback("last", lastCallback);
//    cv::waitKey(0);
}

int main()
{
    rectify("Parallel_Cube");
    rectify("Parallel_Real");
    rectify("Turned_Cube");
    rectify("Turned_Real");

    return 0;
}
