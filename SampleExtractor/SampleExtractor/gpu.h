#pragma once
// modified from sources\samples\gpu\surf_keypoint_matcher.cpp
#include <iostream>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

void testCudaSURF() {
	GpuMat img1;
	Mat img = imread("tmp.bmp", IMREAD_GRAYSCALE);
	Mat dst;
	double scale = 0.8;
	cv::resize(img, dst, Size(), scale, scale, CV_INTER_AREA);

	img1.upload(dst);

	SURF_CUDA surf;

	// detecting keypoints & computing descriptors
	GpuMat keypoints1GPU;
	GpuMat descriptors1GPU;
	surf.hessianThreshold = 16000;
	surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);

	cout << "FOUND " << keypoints1GPU.cols << " keypoints on the image" << endl;

	// downloading results
	vector<KeyPoint> keypoints1;
	surf.downloadKeypoints(keypoints1GPU, keypoints1);

	// drawing the results
	Mat marked;
	drawKeypoints(Mat(img1), keypoints1, marked);
	imwrite("surf points marked.bmp", marked);

	namedWindow("mark", 0);
	imshow("mark", marked);
	waitKey(0);

	img.release();
	dst.release();
	img1.release();
	marked.release();
}

vector<KeyPoint> getCudaSURF(string img_s, int Hes){
	GpuMat img1;
	Mat img = imread(img_s, IMREAD_GRAYSCALE);
	Mat dst;
	double scale = 0.8;
	cv::resize(img, dst, Size(), scale, scale, CV_INTER_AREA);
	// ≤ªresizeœ‘¥Ê≤ªπª£¨¿±º¶1066µƒ6Gœ‘¥Ê
	img1.upload(dst);
	SURF_CUDA surf;

	// detecting keypoints & computing descriptors
	GpuMat keypoints1GPU;
	GpuMat descriptors1GPU;
	surf.hessianThreshold = Hes;
	surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);

	cout << "FOUND " << keypoints1GPU.cols << " keypoints on the image" << endl;

	// downloading results
	vector<KeyPoint> keypoints1;
	surf.downloadKeypoints(keypoints1GPU, keypoints1);

	// drawing the results
	Mat marked;
	drawKeypoints(Mat(img1), keypoints1, marked);
	imwrite("surf points marked.bmp", marked);

	//namedWindow("matches", 0);
	//imshow("matches", img_matches);
	//waitKey(0);
	img.release();
	dst.release();
	img1.release();
	marked.release();

	return keypoints1;
}