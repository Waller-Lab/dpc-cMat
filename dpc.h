
#include <stdio.h>
#include <cmath>
#include <iostream>

#include "../cMat/cMat.cpp"
#include "../cMat/cMat.h"

using namespace cvc;

namespace dpc {

    double[] range(double start, double end, double interval);
    cMat genSrcAngular(cMat srcCoeffs, double rotationAngle, int imSize, double na, double ps, double[] lambdaList);
    cMat[] genHuHp(cMat src, cMat pupil, double[] lambdaList, boolean shiftedOutput);
    cMat genMeasurementsLinear(cMat field, cMat pupil, cMat Hu, cMat Hp, double[] lambdaList);
    cMat genMeasurementsNonlinear(cMat field, cMat src, cMat pupil, double nPositions);
    cMat cdpcLeastSquares(cMat intensity, cMat Hu, cMat Hp, double ampReg, double phaseReg, double resultLambda);

}
