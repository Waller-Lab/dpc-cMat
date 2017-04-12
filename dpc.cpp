#include "dpc.h"

#include <stdio.h>
#include <cmath>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cvc;

/*
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

/*
 * Generates an angular LED source array.
 *
 * @param srcCoeffs                 coefficients to generate the LED pattern
 * @param rotationAngle             constant offset for the LED array sections
 * @param imSize                    [m, n] where the image is an m x n matrix
 * @param ps                        pixel size
 * @param lambdaList                list of lambda values for regularization
 */
cMat dpc::genSrcAngular(cMat srcCoeffs, double rotationAngle, std::vector<int> imSize, double na, double ps, std::vector<double> lambdaList) {
    int sections = sizeof(srcCoeffs);
    double dTheta = 360 / sections;
    cMat angles = dpc::range(rotationAngle, 360 + rotationAngle, dTheta);
    int m = imSize[0];
    int n = imSize[1];
    double dfx = 1 / (n * ps);
    double dfy = 1 / (m * ps);
    cMat fx = dfx * dpc::range(- (n - (n % 2)) / 2, (n - (n % 2)) / 2 - (n % 2 == 1 ? 1 : 0), 1);
    cMat fy = dfx * dpc::range(- (n - (n % 2)) / 2, (n - (n % 2)) / 2 - (n % 2 == 1 ? 1 : 0), 1);
    std::vector<cMat> mesh = cvc::meshgrid(fx, fy);
    cMat fxx = mesh[0];
    cMat fyy = mesh[1];
    cMat srcList = cvc::zeros(m, n);
    //TODO;
    return srcList;
}

/*
 * Generates the transfer function for a source illumination.
 *
 * @param src                       source in real space
 * @param pupil                     pupil function in Fourier space
 * @param lambdaList                list of lambda values for regularization
 * @param shiftedOutput             true if should shift output to origin
 */
std::vector<cMat> dpc::genHuHp(cMat src, cMat pupil, cMat lambdaList, bool shiftedOutput) {
    double dc = (*cvc::sum(cvc::sum((cvc::abs(pupil)^2) * src, 0), 1).get(0, 0)).real();    //TODO axis?
    cMat lambda3 = cvc::reshape(lambdaList, 1);

    cMat sp = src * pupil;
    cMat M = cvc::fft2(sp) * cvc::conj(cvc::fft2(pupil));

    cMat re = cvc::real(M);
    cMat Hu = 2 * (cvc::ifft2(re) / dc) / lambda3;
    std::complex<double> i = * new std::complex<double>(0, 1.0);
    cMat im = i * cvc::imag(M);
    cMat temp = (2 * cvc::ifft2(im) / dc) / lambda3;
    cMat Hp = i * temp;
//    cMat Hp = eye * (2 * cvc::ifft2(im) / dc) / lambda3;

    if (shiftedOutput) {
        cMat uCopy = Hu.copy();
        cMat pCopy = Hp.copy();
        cvc::fftshift(uCopy, Hu);
        cvc::fftshift(pCopy, Hp);
    }
    std::vector<cMat> result (2);
    result[0] = Hu;
    result[1] = Hp;
    return result;
}

/*
 * Generates a linear measurement from complex field at image plane.
 *
 * @param field                     complex illumination field
 * @param pupil                     pupil function in Fourier space
 * @param Hu                        amplitude transfer function
 * @param Hp                        phase transfer function
 * @param lambdaList                lambda values for regularization
 */
cMat dpc::genMeasurementsLinear(cMat field, cMat pupil, cMat Hu, cMat Hp, std::vector<double> lambdaList) {
    cMat intensity = zeros(pupil.size());
    for (int cIdx = 0; cIdx < lambdaList.size(); cIdx++) {

    }

    return intensity;
}

/*******************************************************************************
 ********************************** CPP FILE ***********************************
 ******************************************************************************/

/*
 * TODO
 * Figure out which Mats should be converted to cMats
 * Mat[] and merge/split
 * cv::ellipse and etc.
 */

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

void PupilCompute(cv::Mat& output, double NA, double Lambda, double ps){

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


void SourceCompute(cv::Mat* sourceRgb, double RotAngle, double NA_illum, const double Lambda[3], double ps, double crossOffsetH, double crossOffsetV, const double quadrantCoefficents[][3]) {

    for (int cIdx=0; cIdx<3; cIdx++) {

    // Generate illumination pupil
    cv::Mat pupil = Mat::zeros(sourceRgb[0].rows,sourceRgb[0].cols,sourceRgb[0].type());
    PupilCompute(pupil,NA_illum, Lambda[cIdx], ps);

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
    drawRotatedRect(crossMask, rect1, cv::Scalar(0.0));
    drawRotatedRect(crossMask, rect2, cv::Scalar(0.0));

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

void HrHiCompute(cv::Mat& Hi, cv::Mat& Hr, cv::Mat& Source, cv::Mat& Pupil , double Lambda){


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

void GenerateA(cv::Mat* output, cv::Mat* HrList, cv::Mat* HiList, std::complex<double> Regularization){
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

	for(int16_t cIdx=0; cIdx<3; cIdx++)
	{
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

/*output: array where the output matrix will be stored
*Intensity: measured intensity values, deconstructed in 3 channels?
*A: (A Hermitian)A + Reg
*HrList: vectorized Hr terms of transfer function
*HiList: vectorized Hi terms of transfer function
*Lambda: ???
*/
void ColorDeconvolution_L2(cv::Mat& output, cv::Mat* Intensity, cv::Mat* A, cv::Mat* HrList, cv::Mat* HiList, double Lambda){

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
	exp(S_temp[0], outputS[0]); //???


	complexMultiply(I2, A[0], Ma_temp);
	complexMultiply(I1, A[2], Mb_temp);
	complexDivide((Mb_temp-Ma_temp), A[4], Mc_temp); //Mc_temp has f{absorption}
	ifft2(Mc_temp, Ma_temp); //Ma_temp has absorption
	cv::split(Ma_temp, S_temp);
	outputS[1] = S_temp[0]/Lambda; //normalizing using lambda?

	cv::merge(outputS,2,output);

}
