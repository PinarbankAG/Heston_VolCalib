#include "HestonOptimizer.h"
#include <sstream>
#include <iomanip>

void parseDoubleArray(double * Array, const int Count, const char * String) {
	std::istringstream stream(String);
	for (int i = 0; i < Count; ++i) {
		stream >> Array[i];
	}
}

int main(int argc, char** argv) {

	if (argc < 9) {
		std::cout <<
			"Wrong usage of Heston Calibrator.\nCorrect usage is as follows:\n"
			"HestonCalib.exe [CallsCount] [Maturities] [Strikes] [Market Prices] [Spot] [RiskFree rate] [Forward Curve] [Initial Parameters]\n" << std::endl;
		return -1;
	}

	std::cout << std::setprecision(16);

	int CallsCount = atoi(argv[1]);
	double * MemBlock = MKL_MALLOC(double, 5 * CallsCount);
	double * Maturities = MemBlock;
	double * Strikes = &MemBlock[CallsCount];
	double * MarketPrices = &MemBlock[2 * CallsCount];
	double * r = &MemBlock[3 * CallsCount];
	double * ForwardCurve = &MemBlock[4 * CallsCount];
	double Params[PARAMETERS_COUNT];

	parseDoubleArray(Maturities, CallsCount, argv[2]);
	parseDoubleArray(Strikes, CallsCount, argv[3]);
	parseDoubleArray(MarketPrices, CallsCount, argv[4]);
	double Spot = atof(argv[5]);
	parseDoubleArray(r, CallsCount, argv[6]);
	parseDoubleArray(ForwardCurve, CallsCount, argv[7]);
	parseDoubleArray(Params, PARAMETERS_COUNT, argv[8]);

	int maxIter = 200;
	int trialIter = 100;
	double SolverEpsilon = 1.e-8;

	HestonOptimizer optimizer(maxIter, trialIter, Params);
	optimizer.SetMarketCalib(CallsCount, Maturities, Strikes, MarketPrices, Spot, r, ForwardCurve);
	optimizer.Optimize(SolverEpsilon);
	optimizer.PrintParams();
	mkl_free(MemBlock);

	return 0;
}

