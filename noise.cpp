#include "noise.h"
#include "assert.h"

#include <cmath>
#include <vector>
#define RADIUS 3

void apply_noise(const Mat& src, Mat& dist) {
	cv::Mat noise(src.size(),src.type());
	float m = 0;
	float sigma = 25;
	cv::randn(noise, m, sigma);
	dist = src.clone() + noise;
}

//static double distance(int x1, int y1, int x2, int y2) {
//    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
//}


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


static double* getLinearGaussian(int r, double sigma) {
	assert(sigma > 0);
    double* filter = new double[r];
    for (int i = 0; i < r; i++) {
    	filter[i] = exp(-(i * i) / (2 * sigma * sigma));
//    	std::cout << "filter " << i << "  " << filter[i] << std::endl;
    }

    return filter;
}


void gauss_denoise(const Mat& src, Mat& dist, int r, double sigma) {
	double** filter = getGaussian(r, sigma);
    for (int row = r; row < src.rows - r; row++) {
        for (int column = r; column < src.cols - r; column++) {
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
	double* filterR = getLinearGaussian(256, sigmaR);
	for (int row = r; row < src.rows - r; row++) {
		for (int column = r; column < src.cols - r; column++) {
			double new_pixel_value = 0;
			double weight = 0;
			for (int h = 0; h < r; h++) {
				for (int w = 0; w < r; w++) {
					double gs = filterS[h][w];
					double gr = filterR[abs(src.at<uchar>(row, column) -
							 src.at<uchar>(row + h - r, column + w  - r))];
					new_pixel_value += gs * gr * src.at<uchar>(row + h - r, column + w  - r);
					weight += gr;
				}
			 }

			new_pixel_value = new_pixel_value * r * r;
			new_pixel_value /= weight;
			dist.at<uchar>(row, column) = saturate_cast<uchar>(new_pixel_value);
		}
	}
}


static double vector_diff(std::vector<uchar>* f, std::vector<uchar>* s, int r) {
	double result = 0;
	for (int i = 0; i < r * r; i++) {
		result = result + pow((*f)[i] - (*s)[i], 2);
	}
	return result;
}

static std::vector<std::vector<uchar>*>* get_neighbors_array(const Mat& src, int r) {
	std::vector<std::vector<uchar>*>* result =
			new std::vector<std::vector<uchar>*>(src.rows * src.cols);
	for (int row = r; row < src.rows - r; row++) {
		for (int column = r; column < src.cols - r; column++) {
//			std::cout << "row: " << row << " col: " << column << std::endl;
			(*result)[row * src.cols + column] = new std::vector<uchar>(r * r);
			for (int h = 0; h < r; h++) {
				for (int w = 0; w < r; w++) {
					(*(*result)[row * src.cols + column])[h * r + w]
								= src.at<uchar>(row, column);
				}
			 }
		}
	}

	return result;
}

void nl_means_denoise(const Mat& src, Mat& dist, int r, double sigmaS) {
	double* filterR = getLinearGaussian(256, sigmaS);
	std::vector<std::vector<uchar>*>* neighbors
		= get_neighbors_array(src, r);
	for (int row = r; row < src.rows - r; row++) {
		std::cout << "round " << row << " from " << src.rows << std::endl;
		for (int column = r; column < src.cols - r; column++) {
			double new_pixel_value = 0;
			double weight = 0;
			int amount = 0;
			for (int h = max(r, row - 10); h < min(src.rows - r, row + 10); h++) {
				for (int w = max(r, column - 10); w < min(src.cols - r, column + 10); w++) {
					++amount;
					double vector_difference = vector_diff((*neighbors)[row * src.cols + column],
							(*neighbors)[h * src.cols + w], r);
					double gs = (int)vector_difference / 256 < 255 ? filterR[(int)vector_difference / 256] : 0;
					new_pixel_value += gs * src.at<uchar>(h, w);
					weight += gs;
				}
			 }

			new_pixel_value /= weight;
			dist.at<uchar>(row, column) = saturate_cast<uchar>(new_pixel_value);
		}
	}

	for (auto it = neighbors->begin(); it != neighbors->end(); ++it) {
		delete(*it);
	}

	delete(neighbors);
}

static double find_best_sigma(const Mat& src, const Mat& noised_image,
		std::function<void(const Mat&, Mat&, int, double)> denoiser) {

	return 1.5;
	double best_sigma = 0.1;
	double best_result = DBL_MAX;
	for (double sigma = 0.1; sigma < RADIUS; sigma += 0.5) {
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

static std::pair<double, double> find_best_sigmas(const Mat& src, const Mat& noised_image,
		std::function<void(const Mat&, Mat&, int, double, double)> denoiser) {
	return std::make_pair(4.6, 20);
	double best_sigma1 = 0.1;
	double best_sigma2 = 1;
	double best_result = DBL_MAX;
	for (double sigma1 = 0.1; sigma1 < RADIUS; sigma1 += 0.5) {
		for (double sigma2 = 1; sigma2 < 128; sigma2 += 10) {
			Mat image = noised_image.clone();
			denoiser(noised_image, image, RADIUS, sigma1, sigma2);
			double result = mean_absolute_difference(src, image);
			if (result < best_result) {
				best_result = result;
				best_sigma1 = sigma1;
				best_sigma2 = sigma2;
			}
		}
	}

	return std::make_pair(best_sigma1, best_sigma2);
}


static void find_best_sigmas_and_apply(const Mat& src, const Mat& noised_image, Mat& dist,
		std::function<void(const Mat&, Mat&, int, double, double)> denoiser) {
	std::pair<double, double> best_sigmas = find_best_sigmas(src, noised_image, denoiser);

	std::cout << "Best sigmas:" << best_sigmas.first << " and "
			<< best_sigmas.second << std::endl;

	dist = noised_image.clone();
	denoiser(noised_image, dist, RADIUS, best_sigmas.first, best_sigmas.second);
}

static void find_best_sigmas_and_apply(const Mat& src, const Mat& noised_image, Mat& dist,
		std::function<void(const Mat&, Mat&, int, double)> denoiser) {
	double best_sigma = find_best_sigma(src, noised_image, denoiser);

	std::cout << "Best sigma: " << best_sigma << std::endl;

	dist = noised_image.clone();
	denoiser(noised_image, dist, RADIUS, best_sigma);
}


void gaussian_method(const Mat& src, const Mat& noised_image, Mat& dist) {
	find_best_sigmas_and_apply(src, noised_image, dist, gauss_denoise);
}


void bilateral_method(const Mat& src, const Mat& noised_image, Mat& dist) {
	find_best_sigmas_and_apply(src, noised_image, dist, bilateral_denoise);
}

void nl_means_method(const Mat& src, const Mat& noised_image, Mat& dist) {
	find_best_sigmas_and_apply(src, noised_image, dist, nl_means_denoise);
}
