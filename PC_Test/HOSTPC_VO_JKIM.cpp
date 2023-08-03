#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <iostream>

#include <random>
#define float16_t eigen_broken_float16_t
#undef Success
#include "opencv2/opencv.hpp"
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Core"
#include "eigen/Eigen/SVD"
#undef float16_t
#include <fstream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image_write.h"

// FAST
void FastCornerDetector(const std::vector<std::vector<uint8_t>> &frame, std::vector<std::pair<int, int>> &corners)
{
    int threshold = 25;
    corners.clear();

    int width = frame[0].size();
    int height = frame.size();

    std::vector<std::vector<int>> score(height, std::vector<int>(width, 0));

    for (int y = 3; y < height - 3; y++)
    {
        for (int x = 3; x < width - 3; x++)
        {
            int p = frame[y][x];
            std::array<int, 16> circlePixels = {
                frame[y - 3][x], frame[y - 3][x + 1], frame[y - 2][x + 2], frame[y - 1][x + 3],
                frame[y][x + 3], frame[y + 1][x + 3], frame[y + 2][x + 2], frame[y + 3][x + 1],
                frame[y + 3][x], frame[y + 3][x - 1], frame[y + 2][x - 2], frame[y + 1][x - 3],
                frame[y][x - 3], frame[y - 1][x - 3], frame[y - 2][x - 2], frame[y - 3][x - 1]};

            int count = 0;
            for (size_t i = 0; i < circlePixels.size(); i++)
            {
                if (std::abs(circlePixels[i] - p) > threshold)
                {
                    count++;
                    if (count >= 12)
                    {
                        corners.push_back({x, y});
                        score[y][x] = circlePixels[i] - p;
                        break;
                    }
                }
                else
                {
                    count = 0;
                }
            }
        }
    }
    // non-maximal suppression
    for (int y = 3; y < height - 3; y++)
    {
        for (int x = 3; x < width - 3; x++)
        {
            int s = score[y][x];

            if (s == 0)
                continue;

            if (s > 0)
            {
                if (score[y - 1][x] >= s || score[y + 1][x] >= s ||
                    score[y][x - 1] >= s || score[y][x + 1] >= s)
                {
                    corners.erase(std::remove_if(corners.begin(), corners.end(), [x, y](const auto &c)
                                                 { return c.first == x && c.second == y; }),
                                  corners.end());
                }
            }
            else
            {
                if (score[y - 1][x] <= s || score[y + 1][x] <= s ||
                    score[y][x - 1] <= s || score[y][x + 1] <= s)
                {
                    corners.erase(std::remove_if(corners.begin(), corners.end(), [x, y](const auto &c)
                                                 { return c.first == x && c.second == y; }),
                                  corners.end());
                }
            }
        }
    }
 // std::cout << "FAST Features NUM: " << corners.size() << std::endl;
}

// KLT Tracker
void KLT_Tracker(const cv::Mat &prev_frame, const cv::Mat &curr_frame, std::vector<std::pair<int, int>> &prev_corners, std::vector<std::pair<int, int>> &curr_corners)
{
    int window_size = 2;
    int width = curr_frame.cols;
    int height = curr_frame.rows;

    std::vector<std::pair<int, int>> temp_corners;
    std::vector<double> temp_confidences;

    for (const auto &corner : prev_corners)
    {
        int x = corner.first;
        int y = corner.second;

        if (x < window_size || x >= width - window_size || y < window_size || y >= height - window_size)
        {
            continue;
        }

        double dx = 0, dy = 0;
        std::vector<int> dxx, dyy;

        for (int i = -window_size; i <= window_size; i++)
        {
            for (int j = -window_size; j <= window_size; j++)
            {
                double diff = static_cast<double>(curr_frame.at<uint8_t>(y + j, x + i)) - static_cast<double>(prev_frame.at<uint8_t>(y + j, x + i));
                dx += diff * i;
                dy += diff * j;

                if (i != 0 && j != 0)
                {
                    dxx.push_back(dx);
                    dyy.push_back(dy);
                }
            }
        }

        dx /= (2 * window_size + 1) * (2 * window_size + 1);
        dy /= (2 * window_size + 1) * (2 * window_size + 1);

        double flow_magnitude = std::sqrt(dx * dx + dy * dy);
        double flow_mean = std::accumulate(dxx.begin(), dxx.end(), 0.0) / dxx.size();
        double flow_stddev = std::sqrt(std::accumulate(dxx.begin(), dxx.end(), 0.0, [flow_mean](double sum, int val)
                                                       { return sum + (val - flow_mean) * (val - flow_mean); }) /
                                       dxx.size());

        double intensity_change = std::abs(prev_frame.at<uint8_t>(y, x) - curr_frame.at<uint8_t>(y, x));
        double confidence = flow_magnitude + 1.0 / (flow_stddev + 1e-6) + intensity_change;

        temp_corners.push_back({x + static_cast<int>(dx), y + static_cast<int>(dy)});
        temp_confidences.push_back(confidence);
    }

    curr_corners.clear();
    double confidence_threshold = 20.0; // Increase the confidence threshold
    for (size_t i = 0; i < temp_corners.size(); i++)
    {
        if (temp_confidences[i] > confidence_threshold)
        {
            curr_corners.push_back(temp_corners[i]);
        }
    }
    //std::cout << "Tracked Features NUM: " << curr_corners.size() << std::endl;
}

Eigen::Matrix3d CalculateEssentialMatrix(std::vector<std::pair<int, int>> prev_corners, std::vector<std::pair<int, int>> curr_corners)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        int maxInliers = 0;
        int numIterations = 85; //100
        double threshold_err = 1.0;
        Eigen::Matrix3d bestE;

        // Pair corners for shuffle
        std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> paired_corners;
        for (size_t i = 0; i < prev_corners.size(); i++) {
            paired_corners.push_back(std::make_pair(prev_corners[i], curr_corners[i]));
        }

        // for loop for RANSAC Cal
        for (int i = 0; i < numIterations; i++)
        {
            // Shuffle paired corners
            std::shuffle(paired_corners.begin(), paired_corners.end(), g);

            std::vector<std::pair<int, int>> selectedPrevCorners;
            std::vector<std::pair<int, int>> selectedCurrCorners;
            for (int j = 0; j < 8; ++j) {
                selectedPrevCorners.push_back(paired_corners[j].first);
                selectedCurrCorners.push_back(paired_corners[j].second);
            }

            Eigen::MatrixXd A(selectedPrevCorners.size(), 9);
            for (size_t i = 0; i < selectedPrevCorners.size(); i++)
            {
            	double u1 = selectedPrevCorners[i].first;
				double v1 = selectedPrevCorners[i].second;
				double u2 = selectedCurrCorners[i].first;
				double v2 = selectedCurrCorners[i].second;

				A.row(i) << u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1;
            }
            // SVD and forcing E to satisfy the singularity constraint
			Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

			Eigen::Matrix3d E;
			E << svd.matrixV()(0,8), svd.matrixV()(1,8), svd.matrixV()(2,8),
				 svd.matrixV()(3,8), svd.matrixV()(4,8), svd.matrixV()(5,8),
				 svd.matrixV()(6,8), svd.matrixV()(7,8), svd.matrixV()(8,8);

			Eigen::JacobiSVD<Eigen::Matrix3d> svd_E(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::Matrix3d S = svd_E.singularValues().asDiagonal();
			S(2, 2) = 0.0;
			E = svd_E.matrixU() * S * svd_E.matrixV().transpose();

			// Count inliers
			int numInliers = 0;
			for (size_t j = 0; j < prev_corners.size(); j++)
			{
				Eigen::Vector3d p1(prev_corners[j].first, prev_corners[j].second, 1.0);
				Eigen::Vector3d p2(curr_corners[j].first, curr_corners[j].second, 1.0);

				Eigen::Vector3d Ep1 = E * p1;
				Eigen::Vector3d E_T_p2 = E.transpose() * p2;
				double p2_T_E_p1 = p2.transpose() * Ep1;

				double error = p2_T_E_p1 * p2_T_E_p1 / (Ep1.head<2>().squaredNorm() + E_T_p2.head<2>().squaredNorm());

				if (error < threshold_err)
				{
					numInliers++;
				}
			}

			if (numInliers > maxInliers)
			{
			  maxInliers = numInliers;
			  bestE = E;
			}
		}
		return bestE;
	}
// Print bestE
// std::cout << "Best Essential matrix: \n" << bestE << std::endl;

void decomposeEssentialMatrix(const Eigen::Matrix3d &bestE, Eigen::Matrix3d &R_t0, Eigen::Vector3d &t_t0)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(bestE, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Ensure proper rotation
    if (U.determinant() < 0)
    {
        U.col(2) *= -1;
    }
    if (V.determinant() < 0)
    {
        V.col(2) *= -1;
    }

    Eigen::Matrix3d W;
    W << 0, -1, 0,
        1, 0, 0,
        0, 0, 1;

    Eigen::Matrix3d R = U * W * V.transpose();
    Eigen::Vector3d t = U.col(2);

    // Update R_t0 and t_t0
    t_t0 = t_t0 + R_t0 * t; // R_t0 : Rotation from previous pixel, t_t0 : transformation from previous pixel
    R_t0 = R * R_t0;


    std::cout << "=== Updated Rotation Matrix for iteration R_t0 ===" << std::endl;
    std::cout << "R_t0 : \n" << R_t0 << std::endl;
    std::cout << "=== Updated Translation Vector for iteration t_t0 === " << std::endl;
    std::cout << "t_t0 : \n" << t_t0 << std::endl;
    std::cout << "" << std::endl;
}

std::vector<std::vector<uint8_t>> convertMatToVector(const cv::Mat &frame)
{
    std::vector<std::vector<uint8_t>> result;
    result.reserve(frame.rows);

    for (int i = 0; i < frame.rows; ++i)
    {
        const auto row_ptr = frame.ptr<uint8_t>(i);
        result.emplace_back(row_ptr, row_ptr + frame.cols);
    }

    return result;
}

int main()
{
    // Define file stream for writing
    std::ofstream file("straight_VO5_yesfeatures.csv");

    cv::VideoCapture cap("/mnt/disk1/joonoh/VO5_straight/src/yesfeatrues_10.mp4");

    if (!cap.isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat curr_frame, prev_frame;
    std::vector<std::pair<int, int>> prev_corners;
    std::vector<std::pair<int, int>> curr_corners;

    // Initialize R_t0 and t_t0
    Eigen::Matrix3d R_t0 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_t0 = Eigen::Vector3d::Zero();

    // Create a window
    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    // Resize the window ==> Resizing live VIDEO
    cv::resizeWindow("Video", 640, 480);

    //Initialize the path visualization image ==> FOR VISUALIZATION!
    cv::Mat path = cv::Mat::zeros(800, 1200, CV_8UC3);
    //Initialize the location of the camera in the visualization
    cv::Point2f prev_location(path.cols / 2.0f, path.rows / 2.0f);

    // Define the scaling factor
    double scaling_factor = 0.00688331; // Adjust this value based on your camera and scene
    double visualization_scaling_factor = 0.0005; // Adjust this value for better visualization

    // Initialize the location of the camera in meters
    cv::Point2f total_movement(0, 0);
    
   int frame_counter = 0;

    while (1)
    {
        // Read the next frame from the video
        cap >> curr_frame;
       
         frame_counter++;
        if (curr_frame.empty()) // if the tracked features = 0, Just pass this stage!
            break;

        if (!prev_frame.empty())
        {
            // KLT tracking based on the previous frame and the current frame
            KLT_Tracker(prev_frame, curr_frame, prev_corners, curr_corners);

            // Check if we have corners to work with
            if (!curr_corners.empty())
            {
                // Essential Matrix calculation starts here
                Eigen::Matrix3d essentialMatrix = CalculateEssentialMatrix(prev_corners, curr_corners);

                // Decompose essential matrix
                decomposeEssentialMatrix(essentialMatrix, R_t0, t_t0);

                // Visualize camera motion
                cv::Point2f movement = cv::Point2f(t_t0(0), t_t0(2)) * scaling_factor;
                total_movement += movement;
                cv::Point2f curr_location = prev_location + total_movement;

                // Scale the current location for visualization
                cv::Point2f curr_location_vis = cv::Point2f(total_movement.x * visualization_scaling_factor, total_movement.y * visualization_scaling_factor);

             
                // Write to file
                file << std::fixed << std::setprecision(2) << total_movement.x << ", " << total_movement.y << std::endl;

                std::cout << "curr_location_m.x: " << total_movement.x << " m" << std::endl;
                std::cout << "curr_location_m.y: " << total_movement.y << " m" << std::endl;

                std::cout << "frame_counter: " << frame_counter << std::endl;
                
                // Check boundaries
                curr_location_vis.x = std::min(std::max(0.0f, curr_location_vis.x), path.cols - 1.0f);
                curr_location_vis.y = std::min(std::max(0.0f, curr_location_vis.y), path.rows - 1.0f);

                cv::circle(path, curr_location_vis, 3, CV_RGB(0, 255, 0), -1);
                cv::arrowedLine(path, prev_location, curr_location_vis, CV_RGB(255, 0, 0), 1);
                cv::imshow("Camera Path", path);
                prev_location = curr_location;
            }
        }

        // Fast corner detection for the current frame ==> Chainging Mat to Vectory type!
        auto curr_frame_vec = convertMatToVector(curr_frame);
        FastCornerDetector(curr_frame_vec, curr_corners);

        // Display the current frame ==> This type has to ve MAT type!
        cv::imshow("Video", curr_frame);

        // Save the current frame as the previous frame for the next iteration
        prev_frame = curr_frame.clone();
        prev_corners = curr_corners;

       // cv::waitKey(30);
    }

    cap.release();
    //cv::waitKey(0);

    file.close(); //excel
    return 0;
}

