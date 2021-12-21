
#ifndef NOISE_H_
#define NOISE_H_

#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <vector>
#include <opencv2/core/saturate.hpp>
#include <numeric>

using namespace cv;

void apply_noise(const Mat& src, Mat& dist);
double mean_absolute_difference(const Mat& src, const Mat& dist);

void gauss_denoise(const Mat& src, Mat& dist, int r, double sigma);
void bilateral_denoise(const Mat& src, Mat& dist, int r, double sigmaS, double sigmaR);
void nl_means_denoise(const Mat& src, Mat& dist, int r, double sigma);


std::pair<int, double> find_best_r_and_sigma(const Mat& src,
		std::function<void(const Mat&, Mat&, int, double)> denoiser);


void gaussian_method(const Mat&, const Mat&, Mat&);
void bilateral_method(const Mat&, const Mat&, Mat&);
void nl_means_method(const Mat&, const Mat&, Mat&);
#endif /* NOISE_H_ */
