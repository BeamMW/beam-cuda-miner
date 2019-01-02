#ifndef CUDAHOST_H
#define CUDAHOST_H

#include "beamStratum.h"
#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <string>
#include <cstdint>

struct EquihashContext;

namespace beamMiner {

class CudaHost {
public:
	void startMining(beamStratum *stratum, const std::vector<int32_t> &devices);
private:
	EquihashContext *initializeContext(int cudaDeviceIndex, std::string &deviceName);
	bool copySolutions(EquihashContext *context, uint32_t &solutionCount, uint32_t *solutions);
	void workerLoop(int deviceIndex, int cudaDeviceIndex);

	std::vector<int32_t> devices_;
	beamStratum *stratum_;
	
	std::vector<std::shared_ptr<std::thread>> workers_;
	std::mutex countersMutex_;
	std::vector<uint32_t> counters_;
	std::mutex logMutex_;
};

}

#endif

