#include "dpc.h"

#include <stdio.h>
#include <cmath>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cvc;

namespace dpc {

    /*
     * Returns a range from start to end spaced at interval.
     *
     * @param start                     start value for range
     * @param end                       end value for range
     * @param interval                  size of range's intervals
     */
    cMat range(double start, double end, double interval) {
        int sections = (int) (end - start) / interval;
        int sgn = sections >= 0 ? 1 : -1;
        // row vec
        cMat rng (new cv::Size(sgn * sections, 1));
        for (int i = 0; i < sections * sgn; i++) {
            rng.set(0, i, new std::complex<double>(interval * sgn * i + start, 0));
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
    cMat genSrcAngular(double[][] srcCoeffs, double rotationAngle, int[] imSize, double na, double ps, double[] lambdaList) {
        int sections = sizeof(srcCoeffs);
        double dTheta = 360 / sections;
        double[] angles = range(rotationAngle, 360 + rotationAngle, dTheta);
        int m = imSize[0];
        int n = imSize[1];
        double dfx = 1 / (n * ps);
        double dfy = 1 / (m * ps);
        cMat fx = dfx * range(- (n - (n % 2)) / 2, (n - (n % 2)) / 2 - (n % 2 == 1 ? 1 : 0));
        cMat fy = dfx * range(- (n - (n % 2)) / 2, (n - (n % 2)) / 2 - (n % 2 == 1 ? 1 : 0));
        //fxx, fyy = meshgrid?

        srcList = zeros();
    }

    /*
     * Generates the transfer function for a source illumination.
     *
     * @param src                       source in real space
     * @param pupil                     pupil function in Fourier space
     * @param lambdaList                list of lambda values for regularization
     * @param shiftedOutput             true if should shift output to origin
     */
    cMat[] genHuHp(cMat src, cMat pupil, double[] lambdaList, boolean shiftedOutput) {
        double dc = sum(abs(pupil)^2 * src);    //TODO axis?
        cMat lambda3 = reshape(lambdaList, 1);

        cMat M = fft2(src * pupil) * conj(fft2(pupil));

        cMat Hu = 2 * (ifft2(real(M)) / dc) / lambda3;
        std::complex<double> eye = new std::complex<double>(0, 1.0);
        cMat Hp = 2 * eye * (ifft2(eye * imag(M)) / dc) / lambda3;

        if (shiftedOutput) {
            cMat uCopy = Hu.copy();
            cMat pCopy = Hp.copy();
            fftshift(uCopy, Hu);
            fftshift(pCopy, Hp);
        }
        cMat[] result = new cMat[2];
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
    cMat genMeasurementsLinear(cMat field, cMat pupil, cMat Hu, cMat Hp, double[] lambdaList) {
        cMat intensity = zeros(pupil.size());
    }

}
