#include "noise.h"
#include "assert.h"

#define RADIUS 5

void apply_noise(const Mat& src, Mat& dist) {
	cv::Mat noise(src.size(),src.type());
	float m = 0;
	float sigma = 25;
	cv::randn(noise, m, sigma);
	dist = src.clone() + noise;
}


double mean_absolute_difference(const Mat& src, const Mat& dist) {
	assert(src.rows ==  dist.rows);
	assert(src.cols ==  dist.cols);

	int result = 0;

	for (int row = 0; row < src.rows; row++) {
		for (int column = 0; column < src.cols; column++) {
			result = result + abs(src.at<uchar>(row, column) - dist.at<uchar>(row, column));
		}
	}

	return (double)result / (src.rows * src.cols);
}


static double** getGaussian(int r, double sigma) {
	assert(sigma > 0);
    double** filter = new double*[r];
    for (int i = 0; i < r; i++) {
        filter[i] = new double[r];
    }
    double sum = 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
            filter[i][j] = exp(-((i-r) * (i-r) + (j-r) * (j-r)) / (2 * sigma * sigma));
            sum += filter[i][j];
        }
    }

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
            filter[i][j] /= sum;
        }
    }
    return filter;
}


void gauss_denoise(const Mat& src, Mat& dist, int r, double sigma) {
	double** filter = getGaussian(r, sigma);
    for (int row = r; row < src.rows - r; row++) {
        for (int column = r; column < src.cols; column++) {
            double new_pixel_value = 0;
            for (int h = 0; h < r; h++) {
                 for (int w = 0; w < r; w++) {
                	 new_pixel_value += filter[h][w] *
                			 src.at<uchar>(row + h - r, column + w  - r);
                 }
             }
             dist.at<uchar>(row, column) = saturate_cast<uchar>(new_pixel_value);
        }
    }
}


void bilateral_denoise(const Mat& src, Mat& dist, int r, double sigmaS, double sigmaR) {
	double** filterS = getGaussian(r, sigmaS);
	double** filterR = getGaussian(r, sigmaS);
    for (int row = r; row < src.rows - r; row++) {
        for (int column = r; column < src.cols; column++) {
            double new_pixel_value = 0;
            for (int h = 0; h < r; h++) {
                 for (int w = 0; w < r; w++) {
                	 new_pixel_value += filterS[h][w] *
                			 src.at<uchar>(row + h - r, column + w  - r);
                 }
             }
             dist.at<uchar>(row, column) = saturate_cast<uchar>(new_pixel_value);
        }
    }
}

double find_best_sigma(const Mat& src, const Mat& noised_image,
		std::function<void(const Mat&, Mat&, int, double)> denoiser) {
	double best_sigma = 0.1;
	double best_result = DBL_MAX;
	for (double sigma = 0.1; sigma < RADIUS; sigma += 0.1) {
		Mat image = noised_image.clone();
		denoiser(noised_image, image, RADIUS, sigma);
		double result = mean_absolute_difference(src, image);
		if (result < best_result) {
			best_result = result;
			best_sigma = sigma;
		}
	}

	return best_sigma;
}


void find_best_sigma_and_apply(const Mat& src, const Mat& noised_image, Mat& dist,
		std::function<void(const Mat&, Mat&, int, double)> denoiser) {
	double best_sigma = find_best_sigma(src, noised_image, denoiser);

	dist = noised_image.clone();
	denoiser(noised_image, dist, RADIUS, best_sigma);
}


void gaussian_method(const Mat& src, const Mat& noised_image, Mat& dist) {
	find_best_sigma_and_apply(src, noised_image, dist, gauss_denoise);
}
