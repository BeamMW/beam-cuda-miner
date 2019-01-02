#ifndef EQUIHASH150_5_H
#define EQUIHASH150_5_H

static constexpr uint32_t stringCount = 1 << 26;

struct BucketLayout0 {
	static constexpr uint32_t bucketCount = 8192;
	static constexpr uint32_t maxBucketSize = 8672;
	static constexpr uint32_t masking = 8;
	static constexpr uint32_t maxBucketSize2 = 1216;
	static constexpr uint32_t hastTableSize = 512;
};

struct BucketLayout1 {
	static constexpr uint32_t bucketCount = 8192;
	static constexpr uint32_t maxBucketSize = 8672;
	static constexpr uint32_t masking = 4;
	static constexpr uint32_t maxBucketSize2 = 2240;
	static constexpr uint32_t hastTableSize = 1024;
};

struct BucketLayout2 {
	static constexpr uint32_t bucketCount = 8192;
	static constexpr uint32_t maxBucketSize = 8672;
	static constexpr uint32_t masking = 4;
	static constexpr uint32_t maxBucketSize2 = 2240;
	static constexpr uint32_t hastTableSize = 1024;
};

struct BucketLayout3 {
	static constexpr uint32_t bucketCount = 8192;
	static constexpr uint32_t maxBucketSize = 8672;
	static constexpr uint32_t masking = 4;
	static constexpr uint32_t maxBucketSize2 = 2240;
	static constexpr uint32_t hastTableSize = 1024;
};

struct BucketLayout4 {
	static constexpr uint32_t bucketCount = 8192;
	static constexpr uint32_t maxBucketSize = 8672;
	static constexpr uint32_t masking = 4;
	static constexpr uint32_t maxBucketSize2 = 2240;
	static constexpr uint32_t hastTableSize = 1024;
	static constexpr uint32_t maxCandidateCount = 1024;
};

struct EquihashContext {
	static constexpr uint32_t maxSolutionCount = 16;

	uint32_t bucketSizes0[BucketLayout0::bucketCount];
	uint32_t bucketSizes1[BucketLayout1::bucketCount];
	uint32_t bucketSizes2[BucketLayout2::bucketCount];
	uint32_t bucketSizes3[BucketLayout3::bucketCount];
	uint32_t bucketSizes4[BucketLayout4::bucketCount];
	union {
		uint8_t buckets0_0[BucketLayout0::bucketCount * BucketLayout0::maxBucketSize * 16];
		uint8_t buckets2[BucketLayout2::bucketCount * BucketLayout2::maxBucketSize * 16];
	};
	union {
		uint8_t buckets0_1[BucketLayout0::bucketCount * BucketLayout0::maxBucketSize * 8];
		uint8_t buckets4[BucketLayout4::bucketCount * BucketLayout4::maxBucketSize * 16];
	};
	union {
		uint8_t buckets1_0[BucketLayout1::bucketCount * BucketLayout1::maxBucketSize * 16];
		uint8_t buckets3[BucketLayout3::bucketCount * BucketLayout3::maxBucketSize * 16];
	};
	uint8_t buckets1_1[BucketLayout1::bucketCount * BucketLayout1::maxBucketSize * 8];
	uint32_t candidateCount;
	uint2 candidates[BucketLayout4::maxCandidateCount];
	uint32_t solutionCount;
	uint32_t solutions[maxSolutionCount * 32];
};

#endif
