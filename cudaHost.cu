#include "cudaHost.h"
#include "Equihash150_5.cu"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

namespace beamMiner {

void CudaHost::startMining(beamStratum *stratum, const std::vector<int32_t> &devices) {
	stratum_ = stratum;	
	devices_ = devices;
	if (devices.empty()) {
		int deviceCount = 0;
		cudaGetDeviceCount(&deviceCount);
		for (int i = 0; i < deviceCount; ++i)
			devices_.push_back(i);
	}
	counters_.resize(devices_.size());
	int deviceIndex = 0;
	for (int cudaDeviceIndex : devices_) {
		std::shared_ptr<std::thread> thread(new std::thread([this, deviceIndex, cudaDeviceIndex]{
			this->workerLoop(deviceIndex, cudaDeviceIndex);		
		}));
		workers_.push_back(thread);
		++deviceIndex;
	}
	std::vector<uint32_t> sols;
	sols.resize(devices_.size());
	for ( ; ; ) {
		std::this_thread::sleep_for(std::chrono::seconds(15));

		{
			std::lock_guard<std::mutex> lock(countersMutex_);
			for (int i = 0; i < devices_.size(); ++i) {			
				sols[i] = counters_[i];
				counters_[i] = 0;
			}
		}

		{
			std::lock_guard<std::mutex> lock(logMutex_);
			std::cout << "Performance: "; 
			for (int i = 0; i < devices_.size(); ++i) {
				uint32_t sol = sols[i];
				std::cout << std::fixed << std::setprecision(2) << (double)sol / 15.0 << " sol/s ";			
				
			}
			std::cout << endl;
		}
	}
}

EquihashContext *CudaHost::initializeContext(int cudaDeviceIndex, std::string &deviceName) {
	cudaError error;
	cudaDeviceProp prop;
	error = cudaGetDeviceProperties(&prop, cudaDeviceIndex);
	if (error != cudaSuccess)
		return nullptr;
	deviceName = prop.name;
	error = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	if (error != cudaSuccess)
		return nullptr;
	error = cudaSetDevice(cudaDeviceIndex);
	if (error != cudaSuccess)
		return nullptr;	
	EquihashContext *context;
	error = cudaMalloc(&context, sizeof(EquihashContext));
	if (error != cudaSuccess)
		return nullptr;	
	error = cudaMemset(context->bucketSizes0, 0, BucketLayout0::bucketCount * sizeof(uint32_t));
	if (error != cudaSuccess)
		return nullptr;	
	error = cudaMemset(context->bucketSizes1, 0, BucketLayout1::bucketCount * sizeof(uint32_t));
	if (error != cudaSuccess)
		return nullptr;	
	error = cudaMemset(context->bucketSizes2, 0, BucketLayout2::bucketCount * sizeof(uint32_t));
	if (error != cudaSuccess)
		return nullptr;	
	error = cudaMemset(context->bucketSizes3, 0, BucketLayout3::bucketCount * sizeof(uint32_t));
	if (error != cudaSuccess)
		return nullptr;	
	error = cudaMemset(context->bucketSizes4, 0, BucketLayout4::bucketCount * sizeof(uint32_t));
	if (error != cudaSuccess)
		return nullptr;	
	return context;
}

bool CudaHost::copySolutions(EquihashContext *context, uint32_t &solutionCount, uint32_t *solutions) {
	cudaError error = cudaMemcpy(&solutionCount, &context->solutionCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
		return false;
	if (solutionCount > EquihashContext::maxSolutionCount)
		solutionCount = EquihashContext::maxSolutionCount;
	error = cudaMemcpy(solutions, context->solutions, solutionCount * 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return (error == cudaSuccess);
}

void CudaHost::processSolutions(beamStratum::WorkDescription wd, uint32_t solutionCount, uint32_t *solutions) {
	for (uint32_t solutionIndex = 0, *solution = solutions; solutionIndex < solutionCount; ++solutionIndex, solution += 32) {
		for (uint32_t level = 0; level < 5; level++) {
			for (uint32_t *p1 = solution, *p2 = p1 + (1 << 5); p1 != p2; p1 += (2 << level))
				sortPair(p1, 1 << level);
		}
		std::vector<uint32_t> indices(solution, solution + 32);
		stratum_->handleSolution(wd, indices);
	}
}

void CudaHost::sortPair(uint32_t *a, uint32_t len) {
	uint32_t need_sorting = 0;
	for (uint32_t *b = a + len, *aEnd = b; a < aEnd; ++a, ++b) {
		if (need_sorting || *a > *b) {
			need_sorting = 1;
			std::swap(*a, *b);
		} else if (*a < *b)
			return;
	}
}

void CudaHost::workerLoop(int deviceIndex, int cudaDeviceIndex) {
	std::string deviceName;
	EquihashContext *context = initializeContext(cudaDeviceIndex, deviceName);
	{
		std::lock_guard<std::mutex> lock(logMutex_);
		if (context == nullptr) {
			std::cout << "Failed initialize GPU" << cudaDeviceIndex << std::endl;
			return;
		}
		else
			std::cout << "Start mining on GPU" << cudaDeviceIndex << ": " << deviceName << std::endl;
	}
	
	std::vector<uint32_t> solutions(EquihashContext::maxSolutionCount * 32);
	for ( ; ; ) {
		if (!stratum_->hasWork()) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			continue;
		}

		beamStratum::WorkDescription wd;
		uint64_t nonce;
		uint64_t blockHeader[4];
		stratum_->getWork(wd, reinterpret_cast<uint8_t *>(blockHeader));
		nonce = wd.nonce;

		round0<<<stringCount / (3 * 256), 256>>>(context, blockHeader[0], blockHeader[1], blockHeader[2], blockHeader[3], nonce);
		round1<<<BucketLayout0::bucketCount * BucketLayout0::masking, 1024>>>(context);
		round2<<<BucketLayout1::bucketCount * BucketLayout1::masking, 1024>>>(context);
		round3<<<BucketLayout2::bucketCount * BucketLayout2::masking, 1024>>>(context);
		round4<<<BucketLayout3::bucketCount * BucketLayout3::masking, 1024>>>(context);
		round5_0<<<BucketLayout4::bucketCount * BucketLayout4::masking, 1024>>>(context);
		round5_1<<<BucketLayout4::maxCandidateCount, 32>>>(context);
		uint32_t solutionCount;
		if (!copySolutions(context, solutionCount, solutions.data())) {
			std::lock_guard<std::mutex> lock(logMutex_);
			std::cout << "Error on GPU" << cudaDeviceIndex << std::endl;
			return;
		}
		processSolutions(wd, solutionCount, solutions.data());
		if (solutionCount >= EquihashContext::maxSolutionCount)
			solutionCount = EquihashContext::maxSolutionCount;
		{
			std::lock_guard<std::mutex> lock(countersMutex_);
			counters_[deviceIndex] += solutionCount;
		}
	}
}

}
