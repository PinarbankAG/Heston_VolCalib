#pragma once

#include <mkl.h>
#include <tbb/tbb.h>
#include <cmath>
#include <iostream>
#include "HestonConstants.h"

#define MKL_MALLOC(Type, Size) ((Type *)mkl_malloc((Size) * sizeof(Type), 64))

class HestonOptimizer {
private:
	int MaxIterations;
	int TrialStepIterations;
	int CallsCount;
	int Size;
	double Spot;
	double * MemoryChunk;
	double Params[PARAMETERS_COUNT];
	double * CallMaturities;
	double * CallStrikes;
	double * CallMarketPrices;
	double * CallPricesError;
	double * Jacobian;
	double * LeftJacobian;
	double * r;
	double * ForwardCurve;
	double * Fwd;
	double * IntegrationLimit;
	double * dU;
	MKL_Complex16 * zU;
	MKL_Complex16 * zU_i;
	MKL_Complex16 * U2iU;
	MKL_Complex16 * U2iU_i;
	MKL_Complex16 * Transform;
	MKL_Complex16 * iut;
	MKL_Complex16 * Inviu;
	MKL_Complex16 * iut_i;
	MKL_Complex16 * Inviu_i;
	MKL_Complex16 * zOnes;
private:
	void ComputeJacobian(const double*);
	void CTORHandler(const int, const int);
	void Compute_d(double const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16*) const;
	void Compute_A(MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16*, MKL_Complex16*) const;
	void Compute_A(MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16*, MKL_Complex16*, MKL_Complex16*, MKL_Complex16*, MKL_Complex16*, MKL_Complex16*) const;
	void Compute_D(double const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16*, MKL_Complex16*) const;
	void Compute_Phi(double const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16*) const;
	void HestonCallPrices(const double*, double*, const bool) const;
	void Compute_h1(MKL_Complex16 const*, MKL_Complex16*) const;
	void Compute_h2(double const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16*, MKL_Complex16*) const;
	void Compute_h3(double const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*,
		MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16*, MKL_Complex16*) const;
	void Compute_h4(double const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*,
		MKL_Complex16*, MKL_Complex16*) const;
	void Compute_h5(double const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*,
		MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16*, MKL_Complex16*) const;
	void Stage1_PartialDerivatives(double const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*,
		MKL_Complex16 const*, MKL_Complex16 const*,
		MKL_Complex16*, MKL_Complex16*, MKL_Complex16*, MKL_Complex16*, MKL_Complex16*, MKL_Complex16*) const;
	void Stage2_PartialDerivatives(double const*, MKL_Complex16 const*, MKL_Complex16 const*,
		MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*,
		MKL_Complex16*, MKL_Complex16*, MKL_Complex16*) const;
	void Stage3_PartialDerivatives(double const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*,
		MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*,
		MKL_Complex16*, MKL_Complex16*, MKL_Complex16*, MKL_Complex16*, MKL_Complex16*) const;
	void Stage4_PartialDerivatives(MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*, MKL_Complex16 const*,
		MKL_Complex16*) const;
public:
	HestonOptimizer(
		const int maxIter,
		const int trialIter,
		const double* InitialGuess
	);
	void SetMarketCalib(
		const int CallCount,
		const double* Maturities,
		const double* Strikes,
		const double* MarketPrices,
		const double _Spot,
		const double * _r,
		const double * _ForwardCurve
	);
	void Optimize(double);
	void PrintParams() const;
	double* GetParams() const;
	~HestonOptimizer();
};

void HestonOptimizer::CTORHandler(const int maxIter, const int trialIter) {
	MaxIterations = maxIter;
	TrialStepIterations = trialIter;
	MemoryChunk = nullptr;
}

HestonOptimizer::HestonOptimizer(const int maxIter, const int trialIter, const double* InitialGuess) {
	CTORHandler(maxIter, trialIter);
	memcpy(Params, InitialGuess, PARAMETERS_COUNT * sizeof(double));
}

HestonOptimizer::~HestonOptimizer() {
	mkl_free(MemoryChunk);
}

void HestonOptimizer::SetMarketCalib(const int CallCount, const double* Maturities, const double* Strikes,
									 const double* MarketPrices, const double _Spot, const double * _r, const double * _ForwardCurve) {
	Spot = _Spot;
	CallsCount = CallCount;
	Size = INTEGRATION_POINTS * CallsCount;
	auto CallsSize = sizeof(double) * CallsCount;

	// Free previously allocated memory 

	mkl_free(MemoryChunk);

	// Allocate new memory

	MemoryChunk = MKL_MALLOC(double, 31 * Size);

	zU = (MKL_Complex16*)&MemoryChunk[0];
	zU_i = (MKL_Complex16*)&MemoryChunk[2 * Size];
	U2iU = (MKL_Complex16*)&MemoryChunk[4 * Size];
	U2iU_i = (MKL_Complex16*)&MemoryChunk[6 * Size];
	Transform = (MKL_Complex16*)&MemoryChunk[8 * Size];
	dU = &MemoryChunk[10 * Size];
	Jacobian = &MemoryChunk[11 * Size];
	LeftJacobian = &MemoryChunk[12 * Size];
	CallPricesError = &MemoryChunk[13 * Size];
	CallMarketPrices = &MemoryChunk[14 * Size];
	CallStrikes = &MemoryChunk[15 * Size];
	CallMaturities = &MemoryChunk[16 * Size];
	r = &MemoryChunk[17 * Size];
	ForwardCurve = &MemoryChunk[18 * Size];
	IntegrationLimit = &MemoryChunk[19 * Size];
	Fwd = &MemoryChunk[20 * Size];
	iut = (MKL_Complex16*)&MemoryChunk[21 * Size];
	Inviu = (MKL_Complex16*)&MemoryChunk[23 * Size];
	zOnes = (MKL_Complex16*)&MemoryChunk[25 * Size];
	iut_i = (MKL_Complex16*)&MemoryChunk[27 * Size];
	Inviu_i = (MKL_Complex16*)&MemoryChunk[29 * Size];

	double* Fwd2 = MKL_MALLOC(double, CallsCount);
	double* Fwd3 = MKL_MALLOC(double, CallsCount);

	// Copy input params into class memory

	memcpy(CallMaturities, Maturities, CallsSize);
	memcpy(CallStrikes, Strikes, CallsSize);
	memcpy(CallMarketPrices, MarketPrices, CallsSize);
	memcpy(r, _r, CallsSize);
	memcpy(ForwardCurve, _ForwardCurve, CallsSize);

	// IntegrationLimit
	vdPowx(CallsCount, CallMaturities, -0.55, IntegrationLimit);
	cblas_dscal(CallsCount, 51., IntegrationLimit, 1);

	// dU
	for (int i = 0; i < CallsCount; ++i) {
		memcpy(&dU[i * INTEGRATION_POINTS], Point, INTEGRATION_POINTS * sizeof(double));
		cblas_dscal(INTEGRATION_POINTS, IntegrationLimit[i], &dU[i * INTEGRATION_POINTS], 1);
	}

	// zU
	memset(zU, 0, Size * sizeof(MKL_Complex16));
	cblas_dcopy(Size, dU, 1, (double*)zU, 2);

	// zU_i
	MKL_Complex16 IM = { 0., 1. };
	MKL_Complex16 minusOne = { -1., 0. };
	memcpy(zU_i, zU, Size * sizeof(MKL_Complex16));
	cblas_zaxpy(Size, &IM, &minusOne, 0, zU_i, 1);

	// U2iU, U2iU_i
	vdMul(2 * Size, (double*)zU, (double*)zU, (double*)U2iU);
	cblas_zaxpy(Size, &IM, zU, 1, U2iU, 1);
	vzConj(Size, U2iU, U2iU_i);

	//zOnes
	tbb::parallel_for(0, Size, [&](int i) {
		zOnes[i] = { 1., 0. };
	});

	//Inviu, Inviu_i
	vzDiv(Size, zOnes, zU, Inviu);
	vzDiv(Size, zOnes, zU_i, Inviu_i);
	MKL_Complex16 minusI = { 0., -1. };
	cblas_zscal(Size, &minusI, Inviu, 1);
	cblas_zscal(Size, &minusI, Inviu_i, 1);

	//iut, iut_i
	memcpy(iut, zU, Size * sizeof(MKL_Complex16));
	memcpy(iut_i, zU_i, Size * sizeof(MKL_Complex16));
	tbb::parallel_for(0, CallsCount, [&](int i) {
		MKL_Complex16 Tmp = { 0., CallMaturities[i] };
		cblas_zscal(INTEGRATION_POINTS, &Tmp, &iut[i * INTEGRATION_POINTS], 1);
		cblas_zscal(INTEGRATION_POINTS, &Tmp, &iut_i[i * INTEGRATION_POINTS], 1);
	}, tbb::affinity_partitioner());

	tbb::parallel_invoke([&]() {
		// Fwd
		vdSub(CallsCount, ForwardCurve, CallStrikes, Fwd);
		vdMul(CallsCount, r, CallMaturities, Fwd2);
		cblas_dscal(CallsCount, -1.0, Fwd2, 1);
		vdExp(CallsCount, Fwd2, Fwd2);
		vdMul(CallsCount, Fwd2, Fwd, Fwd);
		cblas_dscal(CallsCount, 0.5, Fwd, 1);
	}, [&]() {
		// Transform
		tbb::parallel_invoke([&]() {
			memcpy(Transform, zU, Size * sizeof(MKL_Complex16));
			tbb::parallel_for(0, CallsCount, [&](int i) {
				MKL_Complex16 Tmp = { 0., -log(CallStrikes[i]) };
				cblas_zscal(INTEGRATION_POINTS, &Tmp, &Transform[i * INTEGRATION_POINTS], 1);
			}, tbb::affinity_partitioner());
			vzExp(Size, Transform, Transform);
		}, [&]() {
			vdMul(CallsCount, r, CallMaturities, Fwd3);
			cblas_dscal(CallsCount, -1.0, Fwd3, 1);
			vdExp(CallsCount, Fwd3, Fwd3);
		});
		for (int i = 0; i < CallsCount; ++i) {
			cblas_dscal(2 * INTEGRATION_POINTS, Fwd3[i], (double*)&Transform[i * INTEGRATION_POINTS], 1);
		}
		cblas_dscal(2 * Size, 1. / M_PI, (double*)Transform, 1);
		vzDiv(Size, Transform, zU, Transform);
		MKL_Complex16 Alpha = { 0., -1. };
		cblas_zscal(Size, &Alpha, Transform, 1);
	});

	mkl_free(Fwd2);
	mkl_free(Fwd3);
}

void HestonOptimizer::Stage1_PartialDerivatives(double const* _Params, MKL_Complex16 const* _u, MKL_Complex16 const* _U2iU, MKL_Complex16 const* _Xi, MKL_Complex16 const* _d,
												MKL_Complex16 const* _Cosh, MKL_Complex16 const* _Sinh,
												MKL_Complex16* _dA2dP, MKL_Complex16* _dA1dP, MKL_Complex16* _dddP, MKL_Complex16* _minusSigmaiud, MKL_Complex16* _tXi2, MKL_Complex16* Buffer) const {
	vzDiv(Size, _u, _d, _minusSigmaiud);
	MKL_Complex16 Alpha = { 0, -_Params[4] };
	cblas_zscal(Size, &Alpha, _minusSigmaiud, 1);

	memcpy(_tXi2, _Xi, Size * sizeof(MKL_Complex16));
	tbb::parallel_for(0, CallsCount, [&](int i) {
		cblas_dscal(2 * INTEGRATION_POINTS, CallMaturities[i]/2., (double*)&_tXi2[i * INTEGRATION_POINTS], 1);
	}, tbb::affinity_partitioner());

	vzMul(Size, _minusSigmaiud, _Xi, _dddP);

	vzMul(Size, _minusSigmaiud, _U2iU, _dA1dP);
	vzMul(Size, _tXi2, _dA1dP, _dA1dP);
	vzMul(Size, _Cosh, _dA1dP, _dA1dP);

	vzMul(Size, _Xi, _Cosh, _dA2dP);
	vzMul(Size, _d, _Sinh, Buffer);
	vzAdd(Size, _dA2dP, Buffer, _dA2dP);
	vzMul(Size, _minusSigmaiud, _dA2dP, _dA2dP);
	memcpy(Buffer, _tXi2, Size * sizeof(MKL_Complex16));
	double One = 1.0;
	cblas_daxpy(Size, 1.0, &One, 0, (double*)Buffer, 2);
	vzMul(Size, Buffer, _dA2dP, _dA2dP);
}

void HestonOptimizer::Stage2_PartialDerivatives(double const* _Params, MKL_Complex16 const* _u, MKL_Complex16 const* _Xi,
												MKL_Complex16 const* _dA1dP, MKL_Complex16 const* _dA2dP, MKL_Complex16 const* _dddP, MKL_Complex16 const* _InvA2, MKL_Complex16 const* _minusSigmaiud, MKL_Complex16 const* _h1,
												MKL_Complex16* _dAdP, MKL_Complex16* _dddSigma, MKL_Complex16* Buffer) const {
	vzMul(Size, _dA2dP, _h1, _dAdP);
	vzAdd(Size, _dA1dP, _dAdP, _dAdP);
	vzMul(Size, _InvA2, _dAdP, _dAdP);

	vzDiv(Size, zOnes, _Xi, Buffer);
	cblas_dscal(2 * Size, -1.0, (double*)Buffer, 1);
	double One = 1.0;
	cblas_daxpy(Size, _Params[2] / _Params[4], &One, 0, (double*)Buffer, 2);
	vzMul(Size, Buffer, _dddP, _dddSigma);

	vzMul(Size, _u, _minusSigmaiud, Buffer);
	MKL_Complex16 im = { 0., 1. };
	cblas_zaxpy(Size, &im, Buffer, 1, _dddSigma, 1);
}

void HestonOptimizer::Stage3_PartialDerivatives(double const* _Params, MKL_Complex16 const* _U2iU, MKL_Complex16 const* _d, MKL_Complex16 const* _A1,
												MKL_Complex16 const* _dA1dP, MKL_Complex16 const* _dA2dP, MKL_Complex16 const* _dddP, MKL_Complex16 const* _InvA2, MKL_Complex16 const* _Inviu, MKL_Complex16 const* _Cosh, MKL_Complex16 const* _tXi2, 
												MKL_Complex16* _dA1dSigma, MKL_Complex16* _dA2dSigma, MKL_Complex16* _dddSigma, MKL_Complex16* _InvBdBdK, MKL_Complex16* Buffer) const {
	tbb::parallel_invoke([&]() {
		vzMul(Size, _U2iU, _Cosh, _dA1dSigma);
		tbb::parallel_for(0, CallsCount, [&](int i) {
			cblas_dscal(2 * INTEGRATION_POINTS, CallMaturities[i] / 2., (double*)&_dA1dSigma[i * INTEGRATION_POINTS], 1);
		}, tbb::affinity_partitioner());
		vzMul(Size, _dddSigma, _dA1dSigma, _dA1dSigma);
	}, [&]() {
		vzMul(Size, _d, _dA2dP, _InvBdBdK);
		vzMul(Size, _InvA2, _InvBdBdK, _InvBdBdK);
		cblas_dscal(2 * Size, -1.0, (double*)_InvBdBdK, 1);
		vzAdd(Size, _dddP, _InvBdBdK, _InvBdBdK);
		vzMul(Size, _Inviu, _InvBdBdK, _InvBdBdK);
		vzDiv(Size, _InvBdBdK, _d, _InvBdBdK);
		cblas_dscal(2 * Size, -1.0 / _Params[4], (double*)_InvBdBdK, 1);
		tbb::parallel_for(0, CallsCount, [&](int i) {
			cblas_daxpy(INTEGRATION_POINTS, 0.5, &CallMaturities[i], 0, (double*)&_InvBdBdK[i * INTEGRATION_POINTS], 2);
		}, tbb::affinity_partitioner());

	}, [&]() {
		memcpy(_dA2dSigma, _A1, Size * sizeof(MKL_Complex16));
		tbb::parallel_for(0, CallsCount, [&](int i) {
			cblas_dscal(2 * INTEGRATION_POINTS, _Params[4] * CallMaturities[i] / 2., (double*)&_dA2dSigma[i * INTEGRATION_POINTS], 1);
		}, tbb::affinity_partitioner());
		cblas_daxpy(2 * Size, _Params[2] / _Params[4], (double*)_dA2dP, 1, (double*)_dA2dSigma, 1);
		vzDiv(Size, zOnes, _tXi2, Buffer);
		double One = 1.0;
		cblas_daxpy(Size, 1.0, &One, 0, (double*)Buffer, 2);
		vzMul(Size, _Inviu, Buffer, Buffer);
		vzMul(Size, _dA1dP, Buffer, Buffer);
		cblas_daxpy(2 * Size, -1.0, (double*)Buffer, 1, (double*)_dA2dSigma, 1);
	});
}

void HestonOptimizer::Stage4_PartialDerivatives(MKL_Complex16 const* _dA1dSigma, MKL_Complex16 const* _dA2dSigma, MKL_Complex16 const* _InvA2, MKL_Complex16 const* _h1,
												MKL_Complex16* _dAdSigma) const {
	vzMul(Size, _dA2dSigma, _h1, _dAdSigma);
	vzAdd(Size, _dA1dSigma, _dAdSigma, _dAdSigma);
	vzMul(Size, _InvA2, _dAdSigma, _dAdSigma);
}

void HestonOptimizer::Compute_h1(MKL_Complex16 const* A, MKL_Complex16* h1) const {
	memcpy(h1, A, Size * sizeof(MKL_Complex16));
	cblas_dscal(2 * Size, -1.0, (double*)h1, 1);
}

void HestonOptimizer::Compute_h2(double const* _Params, MKL_Complex16 const* _iut, MKL_Complex16 const* D, MKL_Complex16* Buffer, MKL_Complex16* h2) const {
	memcpy(h2, _iut, Size * sizeof(MKL_Complex16));
	cblas_dscal(2*Size, -_Params[3] * _Params[2] / _Params[4], (double*)h2, 1);
	memcpy(Buffer, D, Size * sizeof(MKL_Complex16));
	cblas_dscal(2*Size, 2. * _Params[3] / (_Params[4] * _Params[4]), (double*)Buffer, 1);
	vzAdd(Size, h2, Buffer, h2);
}

void HestonOptimizer::Compute_h3(double const* _Params, MKL_Complex16 const* _iut, MKL_Complex16 const* dAdP, MKL_Complex16 const* dddP, MKL_Complex16 const* dA2dP,
								 MKL_Complex16 const* Invd, MKL_Complex16 const* InvA2, MKL_Complex16* Buffer, MKL_Complex16* h3) const {
	auto Tmp = Buffer;
	auto Tmp2 = &Buffer[Size];
	auto Tmp3 = &Buffer[2 * Size];
	
	memcpy(h3, _iut, Size * sizeof(MKL_Complex16));
	cblas_dscal(2 * Size, -_Params[3] * _Params[1] / _Params[4], (double*)h3, 1);

	memcpy(Tmp, dAdP, Size * sizeof(MKL_Complex16));
	cblas_dscal(2 * Size, -_Params[0], (double*)Tmp, 1);

	vzMul(Size, dddP, Invd, Tmp2);

	vzMul(Size, dA2dP, InvA2, Tmp3);
	cblas_dscal(2 * Size, -1.0, (double*)Tmp3, 1);

	vzAdd(Size, Tmp3, Tmp2, Tmp2);
	cblas_dscal(2 * Size, 2 * _Params[3] * _Params[1] / (_Params[4] * _Params[4]), (double*)Tmp2, 1);

	vzAdd(Size, Tmp2, Tmp, Tmp);

	vzAdd(Size, Tmp, h3, h3);
}

void HestonOptimizer::Compute_h4(double const* _Params, MKL_Complex16 const* _iut, MKL_Complex16 const* D, MKL_Complex16 const* dAdP, MKL_Complex16 const* _Inviu, MKL_Complex16 const* InvBdBdK,
								 MKL_Complex16* Buffer, MKL_Complex16* h4) const {
	auto Tmp = Buffer;
	auto Tmp2 = &Buffer[Size];
	auto Tmp3 = &Buffer[2 * Size];

	memcpy(h4, _iut, Size * sizeof(MKL_Complex16));
	cblas_dscal(2 * Size, -_Params[1] * _Params[2] / _Params[4], (double*)h4, 1);

	memcpy(Tmp, InvBdBdK, Size * sizeof(MKL_Complex16));
	cblas_dscal(2 * Size, 2. * _Params[3] * _Params[1] / (_Params[4] * _Params[4]), (double*)Tmp, 1);

	memcpy(Tmp2, D, Size * sizeof(MKL_Complex16));
	cblas_dscal(2 * Size, 2. * _Params[1] / (_Params[4] * _Params[4]), (double*)Tmp2, 1);

	vzMul(Size, dAdP, _Inviu, Tmp3);
	cblas_dscal(2 * Size, _Params[0] / _Params[4], (double*)Tmp3, 1);

	vzAdd(Size, Tmp2, Tmp3, Tmp3);
	vzAdd(Size, Tmp, h4, h4);
	vzAdd(Size, Tmp3, h4, h4);
}

void HestonOptimizer::Compute_h5(double const* _Params, MKL_Complex16 const* _iut, MKL_Complex16 const* D, MKL_Complex16 const* dAdSigma, MKL_Complex16 const* dddSigma, MKL_Complex16 const* dA2dSigma,
								 MKL_Complex16 const* Invd, MKL_Complex16 const* InvA2, MKL_Complex16* Buffer, MKL_Complex16* h5) const {
	auto Tmp = Buffer;
	auto Tmp2 = &Buffer[Size];
	auto Tmp3 = &Buffer[2 * Size];
	auto Tmp4 = &Buffer[3 * Size];

	memcpy(h5, _iut, Size * sizeof(MKL_Complex16));
	cblas_dscal(2 * Size, _Params[3] * _Params[1] * _Params[2] / (_Params[4] * _Params[4]), (double*)h5, 1);

	memcpy(Tmp, dAdSigma, Size * sizeof(MKL_Complex16));
	cblas_dscal(2 * Size, -_Params[0], (double*)Tmp, 1);
	
	vzMul(Size, dddSigma, Invd, Tmp2);

	vzMul(Size, dA2dSigma, InvA2, Tmp3);
	cblas_dscal(2 * Size, -1.0, (double*)Tmp3, 1);

	memcpy(Tmp4, D, Size * sizeof(MKL_Complex16));
	cblas_dscal(2 * Size, -4. * _Params[3] * _Params[1] / (_Params[4] * _Params[4] * _Params[4]), (double*)Tmp4, 1);

	vzAdd(Size, Tmp3, Tmp2, Tmp2);
	cblas_dscal(2 * Size, 2. * _Params[3] * _Params[1] / (_Params[4] * _Params[4]), (double*)Tmp2, 1);

	vzAdd(Size, Tmp4, Tmp2, Tmp2);

	vzAdd(Size, Tmp, h5, h5);

	vzAdd(Size, Tmp2, h5, h5);
}

void HestonOptimizer::Compute_d(double const* _Params, MKL_Complex16 const* _Xi, MKL_Complex16 const* _U2iU, MKL_Complex16* _d) const {
	vzMul(Size, _Xi, _Xi, _d);
	cblas_daxpy(2 * Size, _Params[4] * _Params[4], (double*)_U2iU, 1, (double*)_d, 1);
	vzSqrt(Size, _d, _d);
}

void HestonOptimizer::Compute_A(MKL_Complex16 const* _Xi, MKL_Complex16 const* _U2iU, MKL_Complex16 const* _d, MKL_Complex16* _A, MKL_Complex16* Buffer) const {
	MKL_Complex16* DT = &Buffer[0 * Size];
	MKL_Complex16* P2 = &Buffer[1 * Size];
	MKL_Complex16* P3 = &Buffer[2 * Size];
	memcpy(DT, _d, Size * sizeof(MKL_Complex16));
	for (int i = 0; i < CallsCount; ++i) {
		cblas_dscal(2 * INTEGRATION_POINTS, CallMaturities[i] * 0.5, (double*)&DT[i * INTEGRATION_POINTS], 1);
	}
	vmzSinh(Size, DT, _A, VML_EP);
	vzMul(Size, _A, _Xi, P3);
	vzMul(Size, _U2iU, _A, _A);
	vmzCosh(Size, DT, P2, VML_EP);
	vzMul(Size, _d, P2, P2);
	vzAdd(Size, P2, P3, P3);
	vzDiv(Size, _A, P3, _A);
}

void HestonOptimizer::Compute_A(MKL_Complex16 const* _Xi, MKL_Complex16 const* _U2iU, MKL_Complex16 const* _d, MKL_Complex16* _A, MKL_Complex16* Buffer, MKL_Complex16* _Cosh, MKL_Complex16* _Sinh, MKL_Complex16* _A1, MKL_Complex16* _InvA2) const {
	MKL_Complex16* DT = &Buffer[0 * Size];
	MKL_Complex16* P2 = &Buffer[1 * Size];
	MKL_Complex16* P3 = &Buffer[2 * Size];
	memcpy(DT, _d, Size * sizeof(MKL_Complex16));
	for (int i = 0; i < CallsCount; ++i) {
		cblas_dscal(2 * INTEGRATION_POINTS, CallMaturities[i] * 0.5, (double*)&DT[i * INTEGRATION_POINTS], 1);
	}
	vmzSinh(Size, DT, _A, VML_EP);
	memcpy(_Sinh, _A, Size * sizeof(MKL_Complex16));
	vzMul(Size, _A, _Xi, P3);
	vzMul(Size, _U2iU, _A, _A);
	memcpy(_A1, _A, Size * sizeof(MKL_Complex16));
	vmzCosh(Size, DT, P2, VML_EP);
	memcpy(_Cosh, P2, Size * sizeof(MKL_Complex16));
	vzMul(Size, _d, P2, P2);
	vzAdd(Size, P2, P3, P3);
	memcpy(_InvA2, P3, Size * sizeof(MKL_Complex16));
	vzDiv(Size, zOnes, _InvA2, _InvA2);
	vzDiv(Size, _A, P3, _A);
}

void HestonOptimizer::Compute_D(double const* _Params, MKL_Complex16 const* _Xi, MKL_Complex16 const* _d, MKL_Complex16* _D, MKL_Complex16* Buffer) const {
	MKL_Complex16* P1 = &Buffer[0 * Size];
	MKL_Complex16* P2 = &Buffer[1 * Size];
	MKL_Complex16* P3 = &Buffer[2 * Size];
	MKL_Complex16* P4 = &Buffer[3 * Size];

	tbb::parallel_invoke([&]() {
		memcpy(P1, _d, Size * sizeof(MKL_Complex16));
		cblas_daxpy(Size, -1.0, &_Params[3], 0, (double*)P1, 2);
		tbb::parallel_for(0, CallsCount, [&](int i) {
			cblas_dscal(2 * INTEGRATION_POINTS, CallMaturities[i] * (-0.5), (double*)&P1[i * INTEGRATION_POINTS], 1);
		}, tbb::affinity_partitioner());
	}, [&]() {
		memcpy(P4, _d, Size * sizeof(MKL_Complex16));
		tbb::parallel_for(0, CallsCount, [&](int i) {
			cblas_dscal(2 * INTEGRATION_POINTS, -CallMaturities[i], (double*)&P4[i * INTEGRATION_POINTS], 1);
		}, tbb::affinity_partitioner());
		vzExp(Size, P4, P4);
	}, [&]() {
		vzAdd(Size, _d, _Xi, P2);
		cblas_dscal(2 * Size, 0.5, (double*)P2, 1);
		vzSub(Size, P2, _Xi, P3);
	});

	vzLn(Size, _d, _D);
	vzMul(Size, P4, P3, P3);
	vzAdd(Size, P3, P2, P2);
	vzLn(Size, P2, P2);
	vzAdd(Size, _D, P1, _D);
	vzSub(Size, _D, P2, _D);
}

void HestonOptimizer::Compute_Phi(double const* _Params, MKL_Complex16 const* _U, MKL_Complex16 const* _A, MKL_Complex16 const* _D, MKL_Complex16* _Phi) const {
	memcpy(_Phi, _U, Size * sizeof(MKL_Complex16));
	tbb::parallel_for(0, CallsCount, [&](int i) {
		MKL_Complex16 Tmp = { 0., log(ForwardCurve[i]) - (_Params[3] * _Params[1] * _Params[2] / _Params[4]) * CallMaturities[i] };
		cblas_zscal(INTEGRATION_POINTS, &Tmp, &_Phi[i * INTEGRATION_POINTS], 1);
	}, tbb::affinity_partitioner());
	cblas_daxpy(2 * Size, -_Params[0], (double*)_A, 1, (double*)_Phi, 1);
	cblas_daxpy(2 * Size, 2.0 * _Params[3] * _Params[1] / (_Params[4] * _Params[4]), (double*)_D, 1, (double*)_Phi, 1);
	vzExp(Size, _Phi, _Phi);
}

void HestonOptimizer::HestonCallPrices(const double * _Params, double * _CallPricesError, const bool CalculateError = true) const {
	auto Chunk = MKL_MALLOC(double, 37 * Size);
	MKL_Complex16* Xi = (MKL_Complex16*)&Chunk[0 * Size];
	MKL_Complex16* d = (MKL_Complex16*)&Chunk[2 * Size];
	MKL_Complex16* A = (MKL_Complex16*)&Chunk[4 * Size];
	MKL_Complex16* D = (MKL_Complex16*)&Chunk[6 * Size];
	MKL_Complex16* Phi = (MKL_Complex16*)&Chunk[8 * Size];
	MKL_Complex16* Xi_i = (MKL_Complex16*)&Chunk[10 * Size];
	MKL_Complex16* d_i = (MKL_Complex16*)&Chunk[12 * Size];
	MKL_Complex16* A_i = (MKL_Complex16*)&Chunk[14 * Size];
	MKL_Complex16* D_i = (MKL_Complex16*)&Chunk[16 * Size];
	MKL_Complex16* Phi_i = (MKL_Complex16*)&Chunk[18 * Size];
	MKL_Complex16* Buffer = (MKL_Complex16*)&Chunk[20 * Size];
	MKL_Complex16* Buffer_i = (MKL_Complex16*)&Chunk[28 * Size];
	double* dPhi = &Chunk[36 * Size];

	memset(Xi, 0, Size * sizeof(MKL_Complex16));
	cblas_dcopy(Size, dU, 1, &Xi[0].imag, 2);
	cblas_dscal(Size, -_Params[4] * _Params[2], &Xi[0].imag, 2);
	cblas_daxpy(Size, 1.0, &_Params[3], 0, (double*)Xi, 2);
	memcpy(Xi_i, Xi, Size * sizeof(MKL_Complex16));
	double One = 1.0;
	cblas_daxpy(Size, -_Params[2] * _Params[4], &One, 0, (double*)Xi_i, 2);

	tbb::parallel_invoke([&]() {
		// For u
		Compute_d(_Params, Xi, U2iU, d);
		Compute_A(Xi, U2iU, d, A, Buffer);
		Compute_D(_Params, Xi, d, D, Buffer);
		Compute_Phi(_Params, zU, A, D, Phi);
	}, [&]() {
		// For u_i
		Compute_d(_Params, Xi_i, U2iU_i, d_i);
		Compute_A(Xi_i, U2iU_i, d_i, A_i, Buffer_i);
		Compute_D(_Params, Xi_i, d_i, D_i, Buffer_i);
		Compute_Phi(_Params, zU_i, A_i, D_i, Phi_i);
	});

	tbb::parallel_for(0, CallsCount, [&](int i) {
		cblas_dscal(2 * INTEGRATION_POINTS, -CallStrikes[i], (double *)&Phi[i * INTEGRATION_POINTS], 1);
	}, tbb::affinity_partitioner());
	vzAdd(Size, Phi_i, Phi, Phi);
	vzMul(CallsCount * INTEGRATION_POINTS, Phi, Transform, Phi);
	cblas_dcopy(Size, (double*)Phi, 2, dPhi, 1);
	cblas_dgemv(CblasRowMajor, CblasNoTrans, CallsCount, INTEGRATION_POINTS, 1.0, dPhi, INTEGRATION_POINTS, Weight, 1, 0.0, _CallPricesError, 1);

	vdMul(CallsCount, _CallPricesError, IntegrationLimit, _CallPricesError);
	cblas_dscal(CallsCount, 0.5, _CallPricesError, 1);
	vdAdd(CallsCount, Fwd, _CallPricesError, _CallPricesError);
	if (CalculateError) {
		cblas_daxpy(CallsCount, -1.0, CallMarketPrices, 1, _CallPricesError, 1);
	}
	mkl_free(Chunk);
}

void HestonOptimizer::ComputeJacobian(const double * _Params) {
	auto Chunk = MKL_MALLOC(MKL_Complex16, 65 * Size);
	MKL_Complex16* Xi = &Chunk[0 * Size];
	MKL_Complex16* d = &Chunk[1 * Size];
	MKL_Complex16* A = &Chunk[2 * Size];
	MKL_Complex16* D = &Chunk[3 * Size];
	MKL_Complex16* Phi = &Chunk[4 * Size];
	MKL_Complex16* h1 = &Chunk[5 * Size];
	MKL_Complex16* h2 = &Chunk[6 * Size];
	MKL_Complex16* h3 = &Chunk[7 * Size];
	MKL_Complex16* h4 = &Chunk[8 * Size];
	MKL_Complex16* h5 = &Chunk[9 * Size];
	MKL_Complex16* dAdP = &Chunk[10 * Size];
	MKL_Complex16* dddP = &Chunk[11 * Size];
	MKL_Complex16* dA1dP = &Chunk[12 * Size];
	MKL_Complex16* dA2dP = &Chunk[13 * Size];
	MKL_Complex16* Invd = &Chunk[14 * Size];
	MKL_Complex16* InvA2 = &Chunk[15 * Size];
	MKL_Complex16* InvBdBdK = &Chunk[16 * Size];
	MKL_Complex16* dAdSigma = &Chunk[17 * Size];
	MKL_Complex16* dddSigma = &Chunk[18 * Size];
	MKL_Complex16* dA1dSigma = &Chunk[19 * Size];
	MKL_Complex16* dA2dSigma = &Chunk[20 * Size];
	MKL_Complex16* A1 = &Chunk[21 * Size];
	MKL_Complex16* Cosh = &Chunk[22 * Size];
	MKL_Complex16* Sinh = &Chunk[23 * Size];
	MKL_Complex16* tXi2 = &Chunk[24 * Size];
	MKL_Complex16* minusSigmaiud = &Chunk[25 * Size];
	MKL_Complex16* Buffer = &Chunk[26 * Size];
	MKL_Complex16* Xi_i = &Chunk[30 * Size];
	MKL_Complex16* d_i = &Chunk[31 * Size];
	MKL_Complex16* A_i = &Chunk[32 * Size];
	MKL_Complex16* D_i = &Chunk[33 * Size];
	MKL_Complex16* Phi_i = &Chunk[34 * Size];
	MKL_Complex16* h1_i = &Chunk[35 * Size];
	MKL_Complex16* h2_i = &Chunk[36 * Size];
	MKL_Complex16* h3_i = &Chunk[37 * Size];
	MKL_Complex16* h4_i = &Chunk[38 * Size];
	MKL_Complex16* h5_i = &Chunk[39 * Size];
	MKL_Complex16* dAdP_i = &Chunk[40 * Size];
	MKL_Complex16* dddP_i = &Chunk[41 * Size];
	MKL_Complex16* dA1dP_i = &Chunk[42 * Size];
	MKL_Complex16* dA2dP_i = &Chunk[43 * Size];
	MKL_Complex16* Invd_i = &Chunk[44 * Size];
	MKL_Complex16* InvA2_i = &Chunk[45 * Size];
	MKL_Complex16* InvBdBdK_i = &Chunk[46 * Size];
	MKL_Complex16* dAdSigma_i = &Chunk[47 * Size];
	MKL_Complex16* dddSigma_i = &Chunk[48 * Size];
	MKL_Complex16* dA1dSigma_i = &Chunk[49 * Size];
	MKL_Complex16* dA2dSigma_i = &Chunk[50 * Size];
	MKL_Complex16* A1_i = &Chunk[51 * Size];
	MKL_Complex16* Cosh_i = &Chunk[52 * Size];
	MKL_Complex16* Sinh_i = &Chunk[53 * Size];
	MKL_Complex16* tXi2_i = &Chunk[54 * Size];
	MKL_Complex16* minusSigmaiud_i = &Chunk[55 * Size];
	MKL_Complex16* Buffer_i = &Chunk[56 * Size];
	MKL_Complex16* h[] = { h1, h2, h3, h4, h5 };
	MKL_Complex16* h_i[] = { h1_i, h2_i, h3_i, h4_i, h5_i };
	double* dH[] = { (double*)&Chunk[60 * Size], (double*)&Chunk[61 * Size], (double*)&Chunk[62 * Size], (double*)&Chunk[63 * Size], (double*)&Chunk[64 * Size] };

	memset(Xi, 0, Size * sizeof(MKL_Complex16));
	cblas_dcopy(Size, dU, 1, &Xi[0].imag, 2);
	cblas_dscal(Size, -_Params[4] * _Params[2], &Xi[0].imag, 2);
	cblas_daxpy(Size, 1.0, &_Params[3], 0, (double*)Xi, 2);
	memcpy(Xi_i, Xi, Size * sizeof(MKL_Complex16));
	double One = 1.0;
	cblas_daxpy(Size, -_Params[2] * _Params[4], &One, 0, (double*)Xi_i, 2);
	tbb::parallel_invoke([&]() {
		Compute_d(_Params, Xi, U2iU, d);
		Compute_A(Xi, U2iU, d, A, Buffer, Cosh, Sinh, A1, InvA2);
		Compute_D(_Params, Xi, d, D, Buffer);
		Compute_Phi(_Params, zU, A, D, Phi);
		vzDiv(Size, zOnes, d, Invd);
		Compute_h1(A, h1);
		Stage1_PartialDerivatives(_Params, zU, U2iU, Xi, d, Cosh, Sinh, dA2dP, dA1dP, dddP, minusSigmaiud, tXi2, Buffer);
		Stage2_PartialDerivatives(_Params, zU, Xi, dA1dP, dA2dP, dddP, InvA2, minusSigmaiud, h1, dAdP, dddSigma, Buffer);
		Stage3_PartialDerivatives(_Params, U2iU, d, A1, dA1dP, dA2dP, dddP, InvA2, Inviu, Cosh, tXi2, dA1dSigma, dA2dSigma, dddSigma, InvBdBdK, Buffer);
		Stage4_PartialDerivatives(dA1dSigma, dA2dSigma, InvA2, h1, dAdSigma);
		Compute_h3(_Params, iut, dAdP, dddP, dA2dP, Invd, InvA2, Buffer, h3);
		Compute_h2(_Params, iut, D, Buffer, h2);
		Compute_h4(_Params, iut, D, dAdP, Inviu, InvBdBdK, Buffer, h4);
		Compute_h5(_Params, iut, D, dAdSigma, dddSigma, dA2dSigma, Invd, InvA2, Buffer, h5);
	}, [&]() {
		Compute_d(_Params, Xi_i, U2iU_i, d_i);
		Compute_A(Xi_i, U2iU_i, d_i, A_i, Buffer_i, Cosh_i, Sinh_i, A1_i, InvA2_i);
		Compute_D(_Params, Xi_i, d_i, D_i, Buffer_i);
		Compute_Phi(_Params, zU_i, A_i, D_i, Phi_i);
		vzDiv(Size, zOnes, d_i, Invd_i);
		Compute_h1(A_i, h1_i);
		Stage1_PartialDerivatives(_Params, zU_i, U2iU_i, Xi_i, d_i, Cosh_i, Sinh_i, dA2dP_i, dA1dP_i, dddP_i, minusSigmaiud_i, tXi2_i, Buffer_i);
		Stage2_PartialDerivatives(_Params, zU_i, Xi_i, dA1dP_i, dA2dP_i, dddP_i, InvA2_i, minusSigmaiud_i, h1_i, dAdP_i, dddSigma_i, Buffer_i);
		Stage3_PartialDerivatives(_Params, U2iU_i, d_i, A1_i, dA1dP_i, dA2dP_i, dddP_i, InvA2_i, Inviu_i, Cosh_i, tXi2_i, dA1dSigma_i, dA2dSigma_i, dddSigma_i, InvBdBdK_i, Buffer_i);
		Stage4_PartialDerivatives(dA1dSigma_i, dA2dSigma_i, InvA2_i, h1_i, dAdSigma_i);
		Compute_h3(_Params, iut_i, dAdP_i, dddP_i, dA2dP_i, Invd_i, InvA2_i, Buffer_i, h3_i);
		Compute_h2(_Params, iut_i, D_i, Buffer_i, h2_i);
		Compute_h4(_Params, iut_i, D_i, dAdP_i, Inviu_i, InvBdBdK_i, Buffer_i, h4_i);
		Compute_h5(_Params, iut_i, D_i, dAdSigma_i, dddSigma_i, dA2dSigma_i, Invd_i, InvA2_i, Buffer_i, h5_i);
	});

	tbb::parallel_for(0, PARAMETERS_COUNT, [&](int i) {
		auto JacobianRow = &Jacobian[i * CallsCount];
		vzMul(Size, Phi, h[i], h[i]);
		vzMul(Size, Phi_i, h_i[i], h_i[i]);
		tbb::parallel_for(0, CallsCount, [&](int j) {
			cblas_dscal(2 * INTEGRATION_POINTS, -CallStrikes[j], (double *)(&h[i][j * INTEGRATION_POINTS]), 1);
		}, tbb::affinity_partitioner());
		vzAdd(Size, h_i[i], h[i], h[i]);
		vzMul(Size, h[i], Transform, h[i]);
		cblas_dcopy(Size, (double*)h[i], 2, dH[i], 1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans, CallsCount, INTEGRATION_POINTS, 1.0, dH[i], INTEGRATION_POINTS, Weight, 1, 0.0, JacobianRow, 1);
		vdMul(CallsCount, JacobianRow, IntegrationLimit, JacobianRow);
		cblas_dscal(CallsCount, 0.5, JacobianRow, 1);
	}, tbb::affinity_partitioner());

	mkl_free(Chunk);
}

/* Non Linear Least Squares Optimizer */
void HestonOptimizer::Optimize(double SolverPrecision) {
	/* Precisions for stop-criteria. */
	double SolverEpsilon[6] = { SolverPrecision,SolverPrecision,SolverPrecision,SolverPrecision,SolverPrecision,SolverPrecision };
	double TrustRegionArea = 0.1;
	auto INF = DBL_MAX;
	double LowerBounds[PARAMETERS_COUNT] = { 0., 0., -1., 0., 0. };
	double UpperBounds[PARAMETERS_COUNT] = { INF, INF, 1., INF, INF };

	/* Initialize solver (allocate memory, set initial values, ...) */
	int RCI_Request = 0;
	_TRNSP_HANDLE_t handle;
	int info[6];
	memset(CallPricesError, 0, CallsCount * sizeof(double));
	memset(Jacobian, 0, CallsCount * PARAMETERS_COUNT * sizeof(double));
	auto ParamsCnt = PARAMETERS_COUNT;
	if (dtrnlspbc_init(&handle, &ParamsCnt, &CallsCount, Params, LowerBounds, UpperBounds, SolverEpsilon, &MaxIterations, &TrialStepIterations, &TrustRegionArea) != TR_SUCCESS) {
		std::cout << "Error in Solver initialization.\n";
		MKL_Free_Buffers();
		return;
	}

	/* Checks the correctness of handle and arrays containing Jacobian matrix,
	objective function, lower and upper bounds, and stopping criteria. */
	if (dtrnlspbc_check(&handle, &ParamsCnt, &CallsCount, Jacobian, CallPricesError, LowerBounds, UpperBounds, SolverEpsilon, info) != TR_SUCCESS) {
		std::cout << "Please check the input parameters.\n";
		MKL_Free_Buffers();
		return;
	}

	/* RCI cycle. */
	do
	{
		if (dtrnlspbc_solve(&handle, CallPricesError, Jacobian, &RCI_Request) != TR_SUCCESS) {
			std::cout << "Error in Solver.\n";
			MKL_Free_Buffers();
			return;
		}
		if (RCI_Request == 1) {
			HestonCallPrices(Params, CallPricesError);
		}
		if (RCI_Request == 2) {
			ComputeJacobian(Params);
		}
	} while (RCI_Request > -1 || RCI_Request < -6);
	dtrnlspbc_delete(&handle);
	MKL_Free_Buffers();
}

/* Print parameters. (in order: V0, Theta, Rho, Kappa, Eta) */
void HestonOptimizer::PrintParams() const {
	std::cout << "V0\t" << Params[0] << "\n";
	std::cout << "Theta\t" << Params[1] << "\n";
	std::cout << "Rho\t" << Params[2] << "\n";
	std::cout << "Kappa\t" << Params[3] << "\n";
	std::cout << "Eta\t" << Params[4] << "\n";
	HestonCallPrices(Params, CallPricesError, false);
	for(int i = 0; i < CallsCount; ++i)
	std::cout << CallPricesError[i] << "\n";
}

double* HestonOptimizer::GetParams() const {
	return (double*)Params;
}
