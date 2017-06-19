#include "dpc.h"

#include <stdio.h>
#include <cmath>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>

//for testing
//#include "opencv2/core.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/contrib.hpp"
//#include <iostream>
#include <string>
//#include <stdio.h>
#include <dirent.h>
#include <fstream>
#include <vector>
#include <complex.h>
#include "../cvComplex/cvComplex.h"
#include "../cvComplex/cvComplex.cpp"
#include "include/json.h"
//#include "cDPC_main.h"
//#include <cmath>
#include <time.h>
//#include "../cvComplex/cvComplex.h"
std::complex<double> ii = std::complex<double>(0,1);

using namespace cv;
using namespace std;

/**
 * Returns a range from start to end spaced at interval.
 *
 * @param start                     start value for range
 * @param end                       end value for range
 * @param interval                  size of range's intervals
 */
cMat dpc::range(double start, double end, double interval) {
    int sections = (int) (end - start) / interval;
    int sgn = sections >= 0 ? 1 : -1;
    // row vec
    cMat rng (* new cv::Size(sgn * sections, 1), 0);
    for (int i = 0; i < sections * sgn; i++) {
        rng.set(0, i, * new std::complex<double>(interval * sgn * i + start, 0));
    }
    return rng;
}

//IntensityC: probaly
void Raw2Color(const cv::Mat& IntensityC, cv::Mat* output){
    if (output[0].empty()){
        output[0] = cv::Mat::zeros(IntensityC.rows/2, IntensityC.cols/2, CV_64FC1);
        output[1] = cv::Mat::zeros(IntensityC.rows/2, IntensityC.cols/2, CV_64FC1);
        output[2] = cv::Mat::zeros(IntensityC.rows/2, IntensityC.cols/2, CV_64FC1);
    }

    double meanR = 0;
    double meanG = 0;
    double meanB = 0;

    for(int i = 0; i < output[0].rows; i++) {
        const double* m_i1 = IntensityC.ptr<double>(2*i);  // Input
        const double* m_i2 = IntensityC.ptr<double>(2*i+1);
        double* o_i1 = output[0].ptr<double>(i);   // Output1
        double* o_i2 = output[1].ptr<double>(i);   // Output2
        double* o_i3 = output[2].ptr<double>(i);   // Output3
        for(int j = 0; j < output[0].cols; j++) {
            o_i1[j] = m_i1[2*j];
            o_i2[j] = ((double) m_i1[2*j+1]+ (double)m_i2[2*j])/2.0;
            o_i3[j] = m_i2[2*j+1];
            meanR += o_i1[j];
            meanG += o_i2[j];
            meanB += o_i3[j];
        }
    }
    meanR = meanR/(output[0].rows*output[0].cols);
    meanG = meanG/(output[0].rows*output[0].cols);
    meanB = meanB/(output[0].rows*output[0].cols);
    output[0] = (output[0]-meanR)/meanR;
    output[1] = (output[1]-meanG)/meanG;
    output[2] = (output[2]-meanB)/meanB;
}

// take I_0 / mean(I_0) - 1.0 to preprocess image
void Normalize(cvc::cMat& output, cvc::cMat& input) {
    std::complex<double> s = cvc::sum(input);
    double mean = s.real() / input.size().area();
    output = input / mean - 1.0;
}

void PupilComputeOld(cv::Mat& output, double NA, double Lambda, double ps){

    //TODO figure out what to do with Mat[] and merge
    Mat planes[] = {Mat::zeros(output.rows, output.cols, CV_64FC1),
        Mat::zeros(output.rows, output.cols, CV_64FC1)};
	planes[0] = Mat::zeros(output.rows, output.cols, CV_64FC1);
    planes[1] = Mat::zeros(output.rows, output.cols, CV_64FC1);
    cv::Point center(cvRound(output.cols / 2), cvRound(output.rows / 2));
    int16_t naRadius_h = (int16_t)ceil(NA * ps * output.cols / Lambda);
    int16_t naRadius_v = (int16_t)ceil(NA * ps * output.rows / Lambda);

    cv::ellipse(planes[0], center, cv::Size(naRadius_h,naRadius_v), 0, 0, 360, cv::Scalar(1.0), -1, 8, 0);

    // FFTshift the pupil so it is consistant with object FT
    //fftShift(planes[0], planes[0]);

    merge(planes, 2, output);
}

void PupilCompute(cvc::cMat& output, double NA, double Lambda, double ps) {

    int16_t r = output.rows();
    int16_t c = output.cols();

    cvc::cMat planes = cvc::zeros(r, c);
    Mat pupil = Mat::zeros(r, c, CV_64FC1);
    Mat zeros = Mat::zeros(r, c, CV_64FC1);

    cv::Point center(cvRound(c / 2), cvRound(r / 2));
    int16_t naRadius_h = (int16_t)ceil(NA * ps * c / Lambda);
    int16_t naRadius_v = (int16_t)ceil(NA * ps * r / Lambda);

    cv::ellipse(pupil, center, cv::Size(naRadius_h,naRadius_v), 0, 0, 360, cv::Scalar(1.0), -1, 8, 0);

    output.real = pupil.getUMat(cv::ACCESS_RW);
    output.imag = zeros.getUMat(cv::ACCESS_RW);
}

void SourceComputeOld(cv::Mat* sourceRgb, double RotAngle, double NA_illum, const double Lambda[3], double ps,
    double crossOffsetH, double crossOffsetV, const double quadrantCoefficents[][3]) {

    for (int cIdx=0; cIdx<3; cIdx++) {

        // Generate illumination pupil
        cv::Mat pupil = Mat::zeros(sourceRgb[0].rows,sourceRgb[0].cols,sourceRgb[0].type());
        PupilComputeOld(pupil,NA_illum, Lambda[cIdx], ps);

        // Get real part of pupil
        cv::Mat pupilSplit[] = {cv::Mat::zeros(pupil.rows, pupil.cols, CV_64FC1),
                                cv::Mat::zeros(pupil.rows, pupil.cols, CV_64FC1)};

        cv::split(pupil,pupilSplit);

        // Tempory 3-channel array

        // center point
        cv::Point center(cvRound(pupil.cols / 2), cvRound(pupil.rows / 2));

        // generate center cross matrix
        cv::Mat crossMask = Mat::ones(pupil.rows,pupil.cols,CV_64FC1);

        // Cross widths in pixels
        double crossH = (crossOffsetV*ps*pupil.rows)/(2*Lambda[cIdx]); // converts NA to pixels
        double crossV = (crossOffsetH*ps*pupil.cols)/(2*Lambda[cIdx]); // converts NA to pixels

        // Horizontal
        //Point pt1(0,center.y-crossV);
        //Point pt2(pupil.cols,center.y+crossV);
        double rectWidth = sqrt(pupil.cols*pupil.cols+pupil.rows*pupil.rows);
        cv::RotatedRect rect1(center, cv::Size(2*crossH,rectWidth),RotAngle);
        cv::RotatedRect rect2(center, cv::Size(rectWidth,2*crossV),RotAngle);

        //TODO what to do with drawRotatedRect?
        dpc::drawRotatedRectMat(crossMask, rect1, cv::Scalar(0.0));
        dpc::drawRotatedRectMat(crossMask, rect2, cv::Scalar(0.0));

        double circleWidth = cv::min(pupil.rows,pupil.cols);

        Mat q = Mat::zeros(pupil.rows,pupil.cols,CV_64FC1);
        Mat q_temp = Mat::zeros(pupil.rows,pupil.cols,CV_64FC1);

        // Generate 90 degree sources (rotate each quadrant)
        cv::ellipse(q_temp, center, cv::Size(cvRound(circleWidth / 2),
        cvRound(circleWidth / 2)), RotAngle, -180, -90, cv::Scalar(1.0), -1, 8, 0);
        q_temp = q_temp*quadrantCoefficents[0][cIdx];
        q = q + q_temp;

        cv::ellipse(q_temp, center, cv::Size(cvRound(circleWidth / 2),
        cvRound(circleWidth / 2)), RotAngle+90, -180, -90, cv::Scalar(1.0), -1, 8, 0);
        q_temp = q_temp*quadrantCoefficents[1][cIdx];
        q = q + q_temp;

        cv::ellipse(q_temp, center, cv::Size(cvRound(circleWidth/ 2),
        cvRound(circleWidth / 2)), RotAngle+180, -180, -90, cv::Scalar(1.0), -1, 8, 0);
        q_temp = q_temp*quadrantCoefficents[2][cIdx];
        q = q + q_temp;

        cv::ellipse(q_temp, center, cv::Size(cvRound(circleWidth / 2),
        cvRound(circleWidth / 2)), RotAngle+270, -180, -90, cv::Scalar(1.0), -1, 8, 0);
        q_temp = q_temp*quadrantCoefficents[3][cIdx];
        q = q + q_temp;

        Mat complexPlanes[2] = {cv::Mat(q.rows,q.cols,q.type()),
                                cv::Mat(q.rows,q.cols,q.type())};

        // add to rgb source and crop to pupil
        // Mat tmpSource = cv::Mat::zeros(pupil.rows,pupil.cols,pupil.type());
        //Mat tmpSource = ;
        //tmpSource = tmpSource.mul(pupilSplit[0]);
        complexPlanes[0] = crossMask.mul(q).mul(pupilSplit[0]);
        complexPlanes[1] = cv::Mat::zeros(crossMask.rows,crossMask.cols,crossMask.type());
        cv::merge(complexPlanes,2,sourceRgb[cIdx]);
    }
}

void SourceCompute(cvc::cMat* sourceRgb, double RotAngle, double NA_illum, const double Lambda[4],
        double ps, double crossOffsetH, double crossOffsetV, const double quadrantCoefficents[][4]) {

    for (int cIdx=0; cIdx < 4; cIdx++) {

        // Generate illumination pupil
        cvc::cMat pupil = cvc::zeros(sourceRgb[0].rows(),sourceRgb[0].cols());     //,sourceRgb[0].type());
        PupilCompute(pupil,NA_illum, Lambda[cIdx], ps);

        //Find the intersection of the sources to make uniform brightness
        cvc::cMat intersect = pupil.copy();

        cv::Point center(cvRound(pupil.cols() / 2), cvRound(pupil.rows() / 2));

        // generate center cross matrix
        cvc::cMat crossMask = cvc::ones(pupil.rows(), pupil.cols());     //,CV_64FC1);

        // Cross widths in pixels
        double crossH = (crossOffsetV*ps*pupil.rows())/(2*Lambda[cIdx]); // converts NA to pixels
        double crossV = (crossOffsetH*ps*pupil.cols())/(2*Lambda[cIdx]); // converts NA to pixels

        // Horizontal
        double rectWidth = sqrt(pupil.cols() * pupil.cols() + pupil.rows() * pupil.rows());
        cv::RotatedRect rect1(center, cv::Size(2 * crossH, rectWidth), RotAngle);
        cv::RotatedRect rect2(center, cv::Size(rectWidth, 2 * crossV), RotAngle + 90);

        dpc::drawRotatedRect(crossMask, rect1, cv::Scalar(0.0));
        dpc::drawRotatedRect(crossMask, rect2, cv::Scalar(0.0));

        double circleWidth = cv::min(pupil.rows(),pupil.cols());

        Mat q = Mat::zeros(pupil.rows(), pupil.cols(), CV_64FC1);
        Mat q_temp = Mat::zeros(pupil.rows(), pupil.cols(), CV_64FC1);

        // Generate 90 degree sources (rotate each quadrant)
        cv::ellipse(q_temp, center, cv::Size(cvRound(circleWidth / 2), cvRound(circleWidth / 2)),
            RotAngle, -180, -90, cv::Scalar(1.0), -1, 8, 0);
        q_temp = q_temp*quadrantCoefficents[cIdx][0];
        q = q + q_temp;
        if (quadrantCoefficents[cIdx][0] == 1) {
            intersect *= q_temp;
        }

        cv::ellipse(q_temp, center, cv::Size(cvRound(circleWidth / 2),
        cvRound(circleWidth / 2)), RotAngle+90, -180, -90, cv::Scalar(1.0), -1, 8, 0);
        q_temp = q_temp * quadrantCoefficents[cIdx][1];
        q = q + q_temp;
        if (quadrantCoefficents[cIdx][1] == 1) {
            intersect *= q_temp;
        }

        cv::ellipse(q_temp, center, cv::Size(cvRound(circleWidth/ 2),
        cvRound(circleWidth / 2)), RotAngle+180, -180, -90, cv::Scalar(1.0), -1, 8, 0);
        q_temp = q_temp*quadrantCoefficents[cIdx][2];
        q = q + q_temp;
        if (quadrantCoefficents[cIdx][2] == 1) {
            intersect *= q_temp;
        }

        cv::ellipse(q_temp, center, cv::Size(cvRound(circleWidth / 2),
        cvRound(circleWidth / 2)), RotAngle+270, -180, -90, cv::Scalar(1.0), -1, 8, 0);
        q_temp = q_temp*quadrantCoefficents[cIdx][3];
        q = q + q_temp;
        if (quadrantCoefficents[cIdx][3] == 1) {
            intersect *= q_temp;
        }

        cvc::cMat complexPlanes (q.rows, q.cols);
//        complexPlanes = crossMask * q * pupil;
        complexPlanes = q * pupil;

        //Make brightness of the pupil uniform
        complexPlanes -= intersect;

        //Create the cross separating each quadrant in the pupil
//        complexPlanes *= crossmask;

        sourceRgb[cIdx] = complexPlanes;
    }
}

void HrHiComputeOld(cv::Mat& Hi, cv::Mat& Hr, cv::Mat& Source, cv::Mat& Pupil , double Lambda){


	//Compute DC term
	cv::Mat Ma_temp = Mat::zeros(Source.rows,Source.cols,Source.type());
	cv::Mat Mb_temp = Mat::zeros(Source.rows,Source.cols,Source.type());
	cv::Mat Mc_temp = Mat::zeros(Source.rows,Source.cols,Source.type());

	cv::Mat FPS_cFP_RI[] = {cv::Mat::zeros(Source.rows, Source.cols, CV_64FC1),
		                    cv::Mat::zeros(Source.rows, Source.cols, CV_64FC1)};
	cv::Mat H_iterm[] = {cv::Mat::zeros(Source.rows, Source.cols, CV_64FC1),
						  cv::Mat::zeros(Source.rows, Source.cols, CV_64FC1)};

	complexConj(Pupil,Ma_temp);
	complexMultiply(Pupil,Ma_temp,Mb_temp);
	complexMultiply(Source,Mb_temp,Ma_temp);
	double DC =  cv::sum(Ma_temp)[0];

	//Compute F[S*P]*conj(F[P])

	ifftShift(Pupil,Ma_temp);
	fft2(Ma_temp,Mb_temp);
	complexConj(Mb_temp,Mc_temp);

	complexMultiply(Source,Pupil,Ma_temp);
	ifftShift(Ma_temp,Mb_temp);
	fft2(Mb_temp,Ma_temp);
	complexMultiply(Ma_temp,Mc_temp,Mb_temp);
	cv::split(Mb_temp, FPS_cFP_RI);

	//Compute Hr
    H_iterm[0] = 2*FPS_cFP_RI[0]/DC;
	cv::merge(H_iterm,2,Ma_temp);
	ifft2(Ma_temp,Hr);

	//Compute Hi
	H_iterm[0] = H_iterm[1];
	H_iterm[1] = 2*FPS_cFP_RI[1]/DC/Lambda;
	cv::merge(H_iterm,2,Ma_temp);
	ifft2(Ma_temp,Mb_temp);
	complexScalarMultiply(ii,Mb_temp,Hi);
}

void HaHpCompute(cvc::cMat& Ha, cvc::cMat& Hp, cvc::cMat& Source, cvc::cMat& Pupil , double Lambda) {

	//Compute DC term
	cvc::cMat Ma_temp = cvc::zeros(Source.rows(), Source.cols());
	cvc::cMat Mb_temp = cvc::zeros(Source.rows(), Source.cols());
	cvc::cMat Mc_temp = cvc::zeros(Source.rows(), Source.cols());

    cvc::cMat FPS_cFP_RI = cvc::zeros(Source.rows(), Source.cols());
    cvc::cMat H_iterm = cvc::zeros(Source.rows(), Source.cols());

    Ma_temp = Source * Pupil * cvc::conj(Pupil);

    double DC = cv::sum(Ma_temp.real.getMat(cv::ACCESS_RW))[0];

	//Compute F[S*P]*conj(F[P])

	cvc::ifftshift(Pupil,Ma_temp);
	Mb_temp = cvc::fft2(Ma_temp);
	Mc_temp = cvc::conj(Mb_temp);

    Ma_temp = Source * Pupil;
	cvc::ifftshift(Ma_temp,Mb_temp);
	Ma_temp = cvc::fft2(Mb_temp);
    FPS_cFP_RI = Ma_temp * Mc_temp;

	//Compute Ha
    H_iterm.real = (2 * FPS_cFP_RI / DC).real;
    cvc::cMat temp = cvc::ifft2(H_iterm);

    Ha = temp;

	//Compute Hp
    H_iterm.real = H_iterm.imag;
    H_iterm.imag = (2 * FPS_cFP_RI / DC / Lambda).imag;
    Mb_temp = cvc::ifft2(H_iterm);

    Hp = *(new std::complex<double>(0,1)) * Mb_temp;       //TODO make sure the reference is assigned not the value
    //H.imag = Mb_temp.imag;
}

void GenerateAOld(cv::Mat* output, cv::Mat* HrList, cv::Mat* HiList, std::complex<double> Regularization){
    if (output[0].empty()) {
        output[0] = cv::Mat::zeros(HrList[0].rows, HrList[0].cols, CV_64FC2);
		output[1] = cv::Mat::zeros(HrList[0].rows, HrList[0].cols, CV_64FC2);
   		output[2] = cv::Mat::zeros(HrList[0].rows, HrList[0].cols, CV_64FC2);
		output[3] = cv::Mat::zeros(HrList[0].rows, HrList[0].cols, CV_64FC2);
		output[4] = cv::Mat::zeros(HrList[0].rows, HrList[0].cols, CV_64FC2);
     }

 	cv::Mat Ma_temp = cv::Mat::zeros(output[0].rows,output[0].cols,CV_64FC2);
 	cv::Mat Mb_temp = cv::Mat::zeros(output[0].rows,output[0].cols,CV_64FC2);
	cv::Mat S_temp[2] = {cv::Mat::zeros(output[0].rows,output[0].cols,CV_64FC1),
	                     cv::Mat::zeros(output[0].rows,output[0].cols,CV_64FC1)};

	for(int16_t cIdx=0; cIdx<3; cIdx++) {
		complexAbs(HrList[cIdx], Ma_temp);
		complexMultiply(Ma_temp, Ma_temp, Mb_temp);
		output[0] = output[0] + Mb_temp;

		complexConj(HrList[cIdx], Ma_temp);
		complexMultiply(HiList[cIdx], Ma_temp, Mb_temp);
		output[1] = output[1] + Mb_temp;

		complexConj(HiList[cIdx], Ma_temp);
		complexMultiply(HrList[cIdx], Ma_temp, Mb_temp);
		output[2] = output[2] + Mb_temp;

		complexAbs(HiList[cIdx], Ma_temp);
		complexMultiply(Ma_temp, Ma_temp, Mb_temp);
		output[3] = output[3] + Mb_temp;
	}

	cv::split(output[0],S_temp);
	S_temp[0] = S_temp[0] + Regularization.real();
	cv::merge(S_temp,2,output[0]);

	cv::split(output[3],S_temp);
	S_temp[0] = S_temp[0] + Regularization.imag();
	cv::merge(S_temp,2,output[3]);

	complexMultiply(output[0], output[3], Ma_temp);
	complexMultiply(output[1], output[2], Mb_temp);
	output[4] = Mb_temp - Ma_temp;

}

void GenerateA(cvc::cMat* output, cvc::cMat* HaList, cvc::cMat* HpList, std::complex<double> Regularization) {

    cv::Size size = HaList[0].size();
    if (output[0].isEmpty()) {
        output[0] = cvc::zeros(size);
		output[1] = cvc::zeros(size);
   		output[2] = cvc::zeros(size);
		output[3] = cvc::zeros(size);
		output[4] = cvc::zeros(size);
     }

    cvc::cMat Ma_temp = cvc::zeros(output[0].rows(), output[0].cols());
    cvc::cMat Mb_temp = cvc::zeros(output[0].rows(), output[0].cols());
    cvc::cMat S_temp = cvc::zeros(output[0].rows(), output[0].cols());

	for(int16_t cIdx = 0; cIdx < 4; cIdx++) {

        Ma_temp = cvc::abs(HaList[cIdx]);
        Mb_temp = Ma_temp * Ma_temp;
		output[0] = output[0] + Mb_temp;

		Ma_temp = cvc::conj(HaList[cIdx]);
		Mb_temp = HpList[cIdx] * Ma_temp;
		output[1] = output[1] + Mb_temp;

        Ma_temp = cvc::conj(HpList[cIdx]);
        Mb_temp = HaList[cIdx] * Ma_temp;
		output[2] = output[2] + Mb_temp;

        Ma_temp = cvc::abs(HpList[cIdx]);
        Mb_temp = Ma_temp * Ma_temp;
		output[3] = output[3] + Mb_temp;
	}

    S_temp = output[0];

    output[0] += Regularization.real();
    output[3] += Regularization.imag();

    Ma_temp = output[0] * output[3];
    Mb_temp = output[1] * output[2];
	output[4] = Mb_temp - Ma_temp;

}

/*output: array where the output matrix will be stored
*Intensity: measured intensity values, deconstructed in 3 channels?
*A: (A Hermitian)A + Reg
*HrList: vectorized Hr terms of transfer function
*HiList: vectorized Hi terms of transfer function
*Lambda: ???
*/
void ColorDeconvolution_L2Old(cv::Mat& output, cv::Mat* Intensity, cv::Mat* A, cv::Mat* HrList, cv::Mat* HiList, double Lambda) {

	cv::Mat Ma_temp = cv::Mat::zeros(output.rows,output.cols,CV_64FC2);
	cv::Mat Mb_temp = cv::Mat::zeros(output.rows,output.cols,CV_64FC2);
	cv::Mat Mc_temp = cv::Mat::zeros(output.rows,output.cols,CV_64FC2);

	cv::Mat I1 = cv::Mat::zeros(output.rows,output.cols,CV_64FC2);
	cv::Mat I2 = cv::Mat::zeros(output.rows,output.cols,CV_64FC2);

	cv::Mat S_temp[2] = {cv::Mat::zeros(output.rows,output.cols,CV_64FC1),
	                     cv::Mat::zeros(output.rows,output.cols,CV_64FC1)};
	cv::Mat outputS[2] = {cv::Mat::zeros(output.rows,output.cols,CV_64FC1),
				          cv::Mat::zeros(output.rows,output.cols,CV_64FC1)};

	for(int16_t cIdx=0; cIdx<3; cIdx++) //for all 3 color channels
	{
		S_temp[0] = Intensity[cIdx]; //???
		cv::merge(S_temp, 2, Mc_temp); //creates 1 multi-channel array out of several single-channel ones
		//size of S_temp is 2
		fft2(Mc_temp, Ma_temp); //Fourier transform intensities
		complexConj(HrList[cIdx], Mb_temp); //Mb_temp has the A Hermitian matrix
		complexMultiply(Ma_temp, Mb_temp, Mc_temp); //multiply Ma_temp and Mb_temp and store them in Mc_temp
		I1 = I1 + Mc_temp; //each time add result from elementwise multiplication

		cv::merge(S_temp, 2, Mc_temp);
		fft2(Mc_temp, Ma_temp);
		complexConj(HiList[cIdx], Mb_temp);
		complexMultiply(Ma_temp, Mb_temp, Mc_temp); //generate I2
		I2 = I2 + Mc_temp;
	}

	complexMultiply(I1, A[3], Ma_temp);
	complexMultiply(I2, A[1], Mb_temp);
	complexDivide((Mb_temp-Ma_temp), A[4], Mc_temp); //Mc_temp now contains f{phase}
	ifft2(Mc_temp, Ma_temp); //Ma_temp has phase
	cv::split(Ma_temp, S_temp); //split into the 3 color channels
//    showImg(S_temp[0], "S_temp", -1);
	exp(S_temp[0], outputS[0]); //???


	complexMultiply(I2, A[0], Ma_temp);
	complexMultiply(I1, A[2], Mb_temp);
	complexDivide((Mb_temp-Ma_temp), A[4], Mc_temp); //Mc_temp has f{absorption}
	ifft2(Mc_temp, Ma_temp); //Ma_temp has absorption
	cv::split(Ma_temp, S_temp);
	outputS[1] = S_temp[0]/Lambda; //normalizing using lambda?

	cv::merge(outputS,2,output);

}

void ColorDeconvolution_L2(cvc::cMat& output, cvc::cMat* Intensity, cvc::cMat* A, cvc::cMat* HaList,
        cvc::cMat* HpList, double Lambda) {

    cv::Size size = output.size();

	cvc::cMat Ma_temp = cvc::zeros(size);
	cvc::cMat Mb_temp = cvc::zeros(size);
	cvc::cMat Mc_temp = cvc::zeros(size);

	cvc::cMat I1 = cvc::zeros(size);
	cvc::cMat I2 = cvc::zeros(size);

    cvc::cMat S_temp = cvc::zeros(size);

	for(int16_t cIdx=0; cIdx<3; cIdx++) {
        S_temp.real = Intensity[cIdx].real;

        Mc_temp = S_temp;
        Ma_temp = cvc::fft2(Mc_temp);
        Mb_temp = cvc::conj(HaList[cIdx]);
        Mc_temp = Ma_temp * Mc_temp;
        I1 = I1 + Mc_temp;

        Mc_temp = S_temp;
        Ma_temp = fft2(Mc_temp);
        Mb_temp = cvc::conj(HpList[cIdx]);
        Mc_temp = Ma_temp * Mb_temp;
        I2 = I2 + Mc_temp;
	}

    Mc_temp = (I2 * A[1] - I1 * A[3]) / A[4];
    S_temp = cvc::ifft2(Mc_temp);
    output.real = cvc::exp(S_temp).real;

    Mc_temp = (I1 * A[2] - I2 * A[0]) / A[4];
    Ma_temp = cvc::ifft2(Mc_temp);
    S_temp = Ma_temp;
    output.imag = (S_temp / Lambda).real;

}
