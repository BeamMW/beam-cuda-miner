#include "Equihash150_5.h"

__constant__ uint64_t blake_iv[] = {
	0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
	0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
	0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
	0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

__device__ __forceinline__ uint2 operator^ (uint2 a, uint2 b)
{
	return make_uint2(a.x ^ b.x, a.y ^ b.y);
}

__device__ __forceinline__ uint2 ROR2(const uint2 a, const int offset)
{
	uint2 result;
	{
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	return result;
}

__device__ __forceinline__ uint2 SWAPUINT2(uint2 value)
{
	return make_uint2(value.y, value.x);
}

__device__ __forceinline__ uint2 ROR24(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x2107);
	result.y = __byte_perm(a.y, a.x, 0x6543);
	return result;
}

__device__ __forceinline__ uint2 ROR16(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x1076);
	result.y = __byte_perm(a.y, a.x, 0x5432);
	return result;
}

__device__ __forceinline__ void G2(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t x, uint64_t y)
{
	a = a + b + x;
	((uint2*)&d)[0] = SWAPUINT2(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
	c = c + d;
	((uint2*)&b)[0] = ROR24(((uint2*)&b)[0] ^ ((uint2*)&c)[0]);
	a = a + b + y;
	((uint2*)&d)[0] = ROR16(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
	c = c + d;
	((uint2*)&b)[0] = ROR2(((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
}

__global__ void round0(EquihashContext *context, uint64_t blockHeader1, uint64_t blockHeader2, uint64_t blockHeader3, uint64_t blockHeader4, uint64_t nonce) {
	__shared__ uint32_t data[256 * 16];

	const uint32_t blockIndex = blockIdx.x;
	const uint32_t threadIndex = threadIdx.x;
	const uint32_t hashIndex = blockIndex * 256 + threadIndex;

	uint64_t v[16];
	uint64_t m[16];
	uint64_t blakeState[8];
	v[0] = blakeState[0] = blake_iv[0] ^ (0x01010000 | 57);
	v[1] = blakeState[1] = blake_iv[1];
	v[2] = blakeState[2] = blake_iv[2];
	v[3] = blakeState[3] = blake_iv[3];
	v[4] = blakeState[4] = blake_iv[4];
	v[5] = blakeState[5] = blake_iv[5];
	v[6] = blakeState[6] = blake_iv[6] ^ 0x576F502D6D616542ULL; // "Beam-PoW"
	v[7] = blakeState[7] = blake_iv[7] ^ 0x0000000500000096ULL; // 150,5
	v[8] = blake_iv[0];
	v[9] = blake_iv[1];
	v[10] = blake_iv[2];
	v[11] = blake_iv[3];
	v[12] = blake_iv[4] ^ 44;
	v[13] = blake_iv[5];
	v[14] = blake_iv[6] ^ 0xffffffffffffffffULL;
	v[15] = blake_iv[7];
	m[0] = blockHeader1;
	m[1] = blockHeader2;
	m[2] = blockHeader3;
	m[3] = blockHeader4;
	m[4] = nonce;
	m[5] = hashIndex;
	m[6] = 0;
	m[7] = 0;
	m[8] = 0;
	m[9] = 0;
	m[10] = 0;
	m[11] = 0;
	m[12] = 0;
	m[13] = 0;
	m[14] = 0;
	m[15] = 0;

	// round 1
	G2(v[0], v[4], v[8], v[12], m[0], m[1]);
	G2(v[1], v[5], v[9], v[13], m[2], m[3]);
	G2(v[2], v[6], v[10], v[14], m[4], m[5]);
	G2(v[3], v[7], v[11], v[15], m[6], m[7]);
	G2(v[0], v[5], v[10], v[15], m[8], m[9]);
	G2(v[1], v[6], v[11], v[12], m[10], m[11]);
	G2(v[2], v[7], v[8], v[13], m[12], m[13]);
	G2(v[3], v[4], v[9], v[14], m[14], m[15]);
	// round 2
	G2(v[0], v[4], v[8], v[12], m[14], m[10]);
	G2(v[1], v[5], v[9], v[13], m[4], m[8]);
	G2(v[2], v[6], v[10], v[14], m[9], m[15]);
	G2(v[3], v[7], v[11], v[15], m[13], m[6]);
	G2(v[0], v[5], v[10], v[15], m[1], m[12]);
	G2(v[1], v[6], v[11], v[12], m[0], m[2]);
	G2(v[2], v[7], v[8], v[13], m[11], m[7]);
	G2(v[3], v[4], v[9], v[14], m[5], m[3]);
	// round 3
	G2(v[0], v[4], v[8], v[12], m[11], m[8]);
	G2(v[1], v[5], v[9], v[13], m[12], m[0]);
	G2(v[2], v[6], v[10], v[14], m[5], m[2]);
	G2(v[3], v[7], v[11], v[15], m[15], m[13]);
	G2(v[0], v[5], v[10], v[15], m[10], m[14]);
	G2(v[1], v[6], v[11], v[12], m[3], m[6]);
	G2(v[2], v[7], v[8], v[13], m[7], m[1]);
	G2(v[3], v[4], v[9], v[14], m[9], m[4]);
	// round 4
	G2(v[0], v[4], v[8], v[12], m[7], m[9]);
	G2(v[1], v[5], v[9], v[13], m[3], m[1]);
	G2(v[2], v[6], v[10], v[14], m[13], m[12]);
	G2(v[3], v[7], v[11], v[15], m[11], m[14]);
	G2(v[0], v[5], v[10], v[15], m[2], m[6]);
	G2(v[1], v[6], v[11], v[12], m[5], m[10]);
	G2(v[2], v[7], v[8], v[13], m[4], m[0]);
	G2(v[3], v[4], v[9], v[14], m[15], m[8]);
	// round 5
	G2(v[0], v[4], v[8], v[12], m[9], m[0]);
	G2(v[1], v[5], v[9], v[13], m[5], m[7]);
	G2(v[2], v[6], v[10], v[14], m[2], m[4]);
	G2(v[3], v[7], v[11], v[15], m[10], m[15]);
	G2(v[0], v[5], v[10], v[15], m[14], m[1]);
	G2(v[1], v[6], v[11], v[12], m[11], m[12]);
	G2(v[2], v[7], v[8], v[13], m[6], m[8]);
	G2(v[3], v[4], v[9], v[14], m[3], m[13]);
	// round 6
	G2(v[0], v[4], v[8], v[12], m[2], m[12]);
	G2(v[1], v[5], v[9], v[13], m[6], m[10]);
	G2(v[2], v[6], v[10], v[14], m[0], m[11]);
	G2(v[3], v[7], v[11], v[15], m[8], m[3]);
	G2(v[0], v[5], v[10], v[15], m[4], m[13]);
	G2(v[1], v[6], v[11], v[12], m[7], m[5]);
	G2(v[2], v[7], v[8], v[13], m[15], m[14]);
	G2(v[3], v[4], v[9], v[14], m[1], m[9]);
	// round 7
	G2(v[0], v[4], v[8], v[12], m[12], m[5]);
	G2(v[1], v[5], v[9], v[13], m[1], m[15]);
	G2(v[2], v[6], v[10], v[14], m[14], m[13]);
	G2(v[3], v[7], v[11], v[15], m[4], m[10]);
	G2(v[0], v[5], v[10], v[15], m[0], m[7]);
	G2(v[1], v[6], v[11], v[12], m[6], m[3]);
	G2(v[2], v[7], v[8], v[13], m[9], m[2]);
	G2(v[3], v[4], v[9], v[14], m[8], m[11]);
	// round 8
	G2(v[0], v[4], v[8], v[12], m[13], m[11]);
	G2(v[1], v[5], v[9], v[13], m[7], m[14]);
	G2(v[2], v[6], v[10], v[14], m[12], m[1]);
	G2(v[3], v[7], v[11], v[15], m[3], m[9]);
	G2(v[0], v[5], v[10], v[15], m[5], m[0]);
	G2(v[1], v[6], v[11], v[12], m[15], m[4]);
	G2(v[2], v[7], v[8], v[13], m[8], m[6]);
	G2(v[3], v[4], v[9], v[14], m[2], m[10]);
	// round 9
	G2(v[0], v[4], v[8], v[12], m[6], m[15]);
	G2(v[1], v[5], v[9], v[13], m[14], m[9]);
	G2(v[2], v[6], v[10], v[14], m[11], m[3]);
	G2(v[3], v[7], v[11], v[15], m[0], m[8]);
	G2(v[0], v[5], v[10], v[15], m[12], m[2]);
	G2(v[1], v[6], v[11], v[12], m[13], m[7]);
	G2(v[2], v[7], v[8], v[13], m[1], m[4]);
	G2(v[3], v[4], v[9], v[14], m[10], m[5]);
	// round 10
	G2(v[0], v[4], v[8], v[12], m[10], m[2]);
	G2(v[1], v[5], v[9], v[13], m[8], m[4]);
	G2(v[2], v[6], v[10], v[14], m[7], m[6]);
	G2(v[3], v[7], v[11], v[15], m[1], m[5]);
	G2(v[0], v[5], v[10], v[15], m[15], m[11]);
	G2(v[1], v[6], v[11], v[12], m[9], m[14]);
	G2(v[2], v[7], v[8], v[13], m[3], m[12]);
	G2(v[3], v[4], v[9], v[14], m[13], m[0]);
	// round 11
	G2(v[0], v[4], v[8], v[12], m[0], m[1]);
	G2(v[1], v[5], v[9], v[13], m[2], m[3]);
	G2(v[2], v[6], v[10], v[14], m[4], m[5]);
	G2(v[3], v[7], v[11], v[15], m[6], m[7]);
	G2(v[0], v[5], v[10], v[15], m[8], m[9]);
	G2(v[1], v[6], v[11], v[12], m[10], m[11]);
	G2(v[2], v[7], v[8], v[13], m[12], m[13]);
	G2(v[3], v[4], v[9], v[14], m[14], m[15]);
	// round 12
	G2(v[0], v[4], v[8], v[12], m[14], m[10]);
	G2(v[1], v[5], v[9], v[13], m[4], m[8]);
	G2(v[2], v[6], v[10], v[14], m[9], m[15]);
	G2(v[3], v[7], v[11], v[15], m[13], m[6]);
	G2(v[0], v[5], v[10], v[15], m[1], m[12]);
	G2(v[1], v[6], v[11], v[12], m[0], m[2]);
	G2(v[2], v[7], v[8], v[13], m[11], m[7]);
	G2(v[3], v[4], v[9], v[14], m[5], m[3]);

	v[0] ^= blakeState[0] ^ v[8];
	v[1] ^= blakeState[1] ^ v[9];
	v[2] ^= blakeState[2] ^ v[10];
	v[3] ^= blakeState[3] ^ v[11];
	v[4] ^= blakeState[4] ^ v[12];
	v[5] ^= blakeState[5] ^ v[13];
	v[6] ^= blakeState[6] ^ v[14];
	v[7] ^= blakeState[7] ^ v[15];

	for (uint32_t i = 0; i < 8; ++i) {
		data[threadIndex * 16 + i * 2] = v[i];
		data[threadIndex * 16 + i * 2 + 1] = v[i] >> 32;
	}

	__syncthreads();

	uint32_t v32[16];
	for (uint32_t i = 0; i < 16; ++i) {
		v32[i] = 0;
		for (uint32_t j = threadIndex & 0xf0; j <= threadIndex; ++j)
			v32[i] += data[j * 16 + i];
		v32[i] = __byte_perm(v32[i], 0, 0x0123);
	}

	uint4 w0;
	uint2 w1;

	uint32_t stringIndex = blockIndex * 256 + threadIndex;
	stringIndex = (stringIndex << 1) + stringIndex;

	w0.x = v32[0];
	w0.y = v32[1];
	w0.z = v32[2];
	w0.w = v32[3];
	w1.x = v32[4];
	w1.y = stringIndex;

	uint32_t outputBucketIndex = w0.x >> 19;
	uint32_t outputBucketSlotIndex = atomicAdd(context->bucketSizes0 + outputBucketIndex, 1);
	if (outputBucketSlotIndex < BucketLayout0::maxBucketSize) {
		const uint32_t outputRawSlotIndex = outputBucketIndex * BucketLayout0::maxBucketSize + outputBucketSlotIndex;
		*(uint4 *)(context->buckets0_0 + outputRawSlotIndex * 16) = w0;
		*(uint2 *)(context->buckets0_1 + outputRawSlotIndex * 8) = w1;
	}

	w0.x = __byte_perm(v32[4], v32[5], 0x0765);
	w0.y = __byte_perm(v32[5], v32[6], 0x0765);
	w0.z = __byte_perm(v32[6], v32[7], 0x0765);
	w0.w = __byte_perm(v32[7], v32[8], 0x0765);
	w1.x = __byte_perm(v32[8], v32[9], 0x0765);
	w1.y = ++stringIndex;

	outputBucketIndex = w0.x >> 19;
	outputBucketSlotIndex = atomicAdd(context->bucketSizes0 + outputBucketIndex, 1);
	if (outputBucketSlotIndex < BucketLayout0::maxBucketSize) {
		const uint32_t outputRawSlotIndex = outputBucketIndex * BucketLayout0::maxBucketSize + outputBucketSlotIndex;
		*(uint4 *)(context->buckets0_0 + outputRawSlotIndex * 16) = w0;
		*(uint2 *)(context->buckets0_1 + outputRawSlotIndex * 8) = w1;
	}

	w0.x = __byte_perm(v32[9], v32[10], 0x1076);
	w0.y = __byte_perm(v32[10], v32[11], 0x1076);
	w0.z = __byte_perm(v32[11], v32[12], 0x1076);
	w0.w = __byte_perm(v32[12], v32[13], 0x1076);
	w1.x = __byte_perm(v32[13], v32[14], 0x1076);
	w1.y = ++stringIndex;

	outputBucketIndex = w0.x >> 19;
	outputBucketSlotIndex = atomicAdd(context->bucketSizes0 + outputBucketIndex, 1);
	if (outputBucketSlotIndex < BucketLayout0::maxBucketSize) {
		const uint32_t outputRawSlotIndex = outputBucketIndex * BucketLayout0::maxBucketSize + outputBucketSlotIndex;
		*(uint4 *)(context->buckets0_0 + outputRawSlotIndex * 16) = w0;
		*(uint2 *)(context->buckets0_1 + outputRawSlotIndex * 8) = w1;
	}
}

struct Round1Data {
	uint32_t ht[BucketLayout0::hastTableSize];
	uint4 w0[BucketLayout0::maxBucketSize2];
	uint2 w1[BucketLayout0::maxBucketSize2];
	uint16_t lookupTable[BucketLayout0::maxBucketSize2];
	uint32_t localBucketSize;
};

__global__ void round1(EquihashContext *context) {
	__shared__ Round1Data data;

	const uint32_t bucketIndex = blockIdx.x / BucketLayout0::masking;
	const uint32_t mask = blockIdx.x % BucketLayout0::masking;
	const uint32_t threadIndex = threadIdx.x;

	const uint32_t bucketSize = umin(context->bucketSizes0[bucketIndex], BucketLayout0::maxBucketSize);
	if (threadIndex < BucketLayout0::hastTableSize)
		data.ht[threadIndex] = 0xffff;
	if (threadIndex == 0)
		data.localBucketSize = 0;

	__syncthreads();

	for (uint32_t slotIndex = threadIndex; slotIndex < bucketSize; slotIndex += blockDim.x) {
		const uint32_t rawSlotIndex = bucketIndex * BucketLayout0::maxBucketSize + slotIndex;
		uint4 w0 = *(uint4 *)(context->buckets0_0 + rawSlotIndex * 16);
		uint2 w1 = *(uint2 *)(context->buckets0_1 + rawSlotIndex * 8);
		if (((w0.x >> 16) & 7) == mask) {
			const uint32_t localSlotIndex = atomicAdd(&data.localBucketSize, 1);
			if (localSlotIndex < BucketLayout0::maxBucketSize2) {
				const uint32_t collision = (w0.x >> 7) & 511;
				const uint32_t prevLocalSlotIndex = atomicExch(data.ht + collision, localSlotIndex);
				data.w0[localSlotIndex] = w0;
				data.w1[localSlotIndex] = w1;
				data.lookupTable[localSlotIndex] = prevLocalSlotIndex;
			}
		}
	}

	__syncthreads();

	const uint32_t localBucketSize = umin(data.localBucketSize, BucketLayout0::maxBucketSize2);
	for (uint32_t localSlotIndex1 = threadIndex; localSlotIndex1 < localBucketSize; localSlotIndex1 += blockDim.x) {
		uint32_t localSlotIndex2 = data.lookupTable[localSlotIndex1];
		const uint4 w0_1 = data.w0[localSlotIndex1];
		const uint2 w1_1 = data.w1[localSlotIndex1];
		while (localSlotIndex2 != 0xffff) {
			const uint4 w0_2 = data.w0[localSlotIndex2];
			const uint2 w1_2 = data.w1[localSlotIndex2];
			localSlotIndex2 = data.lookupTable[localSlotIndex2];

			uint4 w0;
			w0.x = w0_1.y ^ w0_2.y;
			w0.y = w0_1.z ^ w0_2.z;
			w0.z = w0_1.w ^ w0_2.w;
			w0.w = w1_1.x ^ w1_2.x;

			const uint32_t outputBucketIndex = (((w0_1.x ^ w0_2.x) & 127) << 6) | (w0.x >> 26);
			const uint32_t outputBucketSlotIndex = atomicAdd(context->bucketSizes1 + outputBucketIndex, 1);
			if (outputBucketIndex >= BucketLayout1::maxBucketSize)
				continue;

			uint2 w1;
			w1.x = w1_1.y;
			w1.y = w1_2.y;

			const uint32_t outputRawSlotIndex = outputBucketIndex * BucketLayout1::maxBucketSize + outputBucketSlotIndex;
			*(uint4 *)(context->buckets1_0 + outputRawSlotIndex * 16) = w0;
			*(uint2 *)(context->buckets1_1 + outputRawSlotIndex * 8) = w1;
		}
	}

	if ((mask == 0) && (threadIndex == 0))
		context->bucketSizes4[bucketIndex] = 0;
}

__device__ __forceinline__ void extractIndices1(EquihashContext *context, uint32_t bucketIndex_1, uint32_t slotIndex_1, uint32_t &stringIndex1, uint32_t &stringIndex2) {
	const uint32_t rawSlotIndex = bucketIndex_1 * BucketLayout1::maxBucketSize + slotIndex_1;
	uint2 w1 = *(uint2 *)(context->buckets1_1 + rawSlotIndex * 8);
	stringIndex1 = w1.x;
	stringIndex2 = w1.y;
}

struct Round2Data {
	uint32_t ht[BucketLayout1::hastTableSize];
	uint4 w0[BucketLayout1::maxBucketSize2];
	uint32_t lookupTable[BucketLayout1::maxBucketSize2];
	uint32_t localBucketSize;
};

__global__ void round2(EquihashContext *context) {
	__shared__ Round2Data data;

	const uint32_t bucketIndex = blockIdx.x / BucketLayout1::masking;
	const uint32_t mask = blockIdx.x % BucketLayout1::masking;
	const uint32_t threadIndex = threadIdx.x;

	const uint32_t bucketSize = umin(context->bucketSizes1[bucketIndex], BucketLayout1::maxBucketSize);
	data.ht[threadIndex] = 0xffff;
	if (threadIndex == 0)
		data.localBucketSize = 0;

	__syncthreads();

	for (uint32_t slotIndex = threadIndex; slotIndex < bucketSize; slotIndex += blockDim.x) {
		const uint32_t rawSlotIndex = bucketIndex * BucketLayout1::maxBucketSize + slotIndex;
		uint4 w0 = *(uint4 *)(context->buckets1_0 + rawSlotIndex * 16);
		if (((w0.x >> 24) & 3) == mask) {
			const uint32_t localSlotIndex = atomicAdd(&data.localBucketSize, 1);
			if (localSlotIndex < BucketLayout1::maxBucketSize2) {
				const uint32_t collision = (w0.x >> 14) & 1023;
				const uint32_t prevLocalSlotIndex = atomicExch(data.ht + collision, localSlotIndex);
				data.w0[localSlotIndex] = w0;
				data.lookupTable[localSlotIndex] = __byte_perm(slotIndex, prevLocalSlotIndex, 0x1054);
			}
		}
	}

	__syncthreads();

	const uint32_t localBucketSize = umin(data.localBucketSize, BucketLayout1::maxBucketSize2);
	for (uint32_t localSlotIndex1 = threadIndex; localSlotIndex1 < localBucketSize; localSlotIndex1 += blockDim.x) {
		const uint32_t value1 = data.lookupTable[localSlotIndex1];
		const uint32_t slotIndex1 = value1 >> 16;
		uint32_t localSlotIndex2 = value1 & 0xffff;
		const uint4 w0_1 = data.w0[localSlotIndex1];
		while (localSlotIndex2 != 0xffff) {
			const uint4 w0_2 = data.w0[localSlotIndex2];
			const uint32_t value2 = data.lookupTable[localSlotIndex2];
			const uint32_t slotIndex2 = value2 >> 16;
			localSlotIndex2 = value2 & 0xffff;

			const uint32_t xorWork = w0_1.x ^ w0_2.x;

			uint4 w0;
			w0.x = (xorWork & 1) | (bucketIndex << 1) | (slotIndex1 << 14) | (slotIndex2 << 28);
			w0.y = w0_1.y ^ w0_2.y;
			w0.z = w0_1.z ^ w0_2.z;
			w0.w = ((w0_1.w ^ w0_2.w) & 0xfffffc00) | (slotIndex2 >> 4);

			const uint32_t outputBucketIndex = (xorWork >> 1) & 0x1fff;
			const uint32_t outputBucketSlotIndex = atomicAdd(context->bucketSizes2 + outputBucketIndex, 1);
			if (outputBucketIndex >= BucketLayout2::maxBucketSize)
				continue;

			const uint32_t outputRawSlotIndex = outputBucketIndex * BucketLayout2::maxBucketSize + outputBucketSlotIndex;
			*(uint4 *)(context->buckets2 + outputRawSlotIndex * 16) = w0;
		}
	}

	if ((mask == 0) &&
		(threadIndex == 0))
		context->bucketSizes0[bucketIndex] = 0;
}

__device__ __forceinline__ void extractIndices2(EquihashContext *context, uint32_t bucketIndex_2, uint32_t slotIndex_2, uint32_t &bucketIndex_1, uint32_t &slotIndex1_1, uint32_t &slotIndex2_1) {
	const uint32_t rawSlotIndex = bucketIndex_2 * BucketLayout2::maxBucketSize + slotIndex_2;
	uint4 w0 = *(uint4 *)(context->buckets2 + rawSlotIndex * 16);
	bucketIndex_1 = (w0.x >> 1) & 0x1fff;
	slotIndex1_1 = (w0.x >> 14) & 0x3fff;
	slotIndex2_1 = (w0.x >> 28) | ((w0.w & 0x03ff) << 4);
}

struct Round3Data {
	uint32_t ht[BucketLayout2::hastTableSize];
	uint3 w0[BucketLayout2::maxBucketSize2];
	uint32_t lookupTable[BucketLayout2::maxBucketSize2];
	uint32_t localBucketSize;
};

__global__ void round3(EquihashContext *context) {
	__shared__ Round3Data data;

	const uint32_t bucketIndex = blockIdx.x / BucketLayout2::masking;
	const uint32_t mask = blockIdx.x % BucketLayout2::masking;
	const uint32_t threadIndex = threadIdx.x;

	const uint32_t bucketSize = umin(context->bucketSizes2[bucketIndex], BucketLayout2::maxBucketSize);
	data.ht[threadIndex] = 0xffff;
	if (threadIndex == 0)
		data.localBucketSize = 0;

	__syncthreads();

	for (uint32_t slotIndex = threadIndex; slotIndex < bucketSize; slotIndex += blockDim.x) {
		const uint32_t rawSlotIndex = bucketIndex * BucketLayout2::maxBucketSize + slotIndex;
		uint4 w0 = *(uint4 *)(context->buckets2 + rawSlotIndex * 16);
		if ((((w0.x & 1) << 1) | (w0.y >> 31)) == mask) {
			const uint32_t localSlotIndex = atomicAdd(&data.localBucketSize, 1);
			if (localSlotIndex < BucketLayout2::maxBucketSize2) {
				const uint32_t collision = (w0.y >> 21) & 1023;
				const uint32_t prevLocalSlotIndex = atomicExch(data.ht + collision, localSlotIndex);
				data.w0[localSlotIndex] = make_uint3(w0.y, w0.z, w0.w);
				data.lookupTable[localSlotIndex] = __byte_perm(slotIndex, prevLocalSlotIndex, 0x1054);
			}
		}
	}

	__syncthreads();

	const uint32_t localBucketSize = umin(data.localBucketSize, BucketLayout2::maxBucketSize2);
	for (uint32_t localSlotIndex1 = threadIndex; localSlotIndex1 < localBucketSize; localSlotIndex1 += blockDim.x) {
		const uint32_t value1 = data.lookupTable[localSlotIndex1];
		const uint32_t slotIndex1 = value1 >> 16;
		uint32_t localSlotIndex2 = value1 & 0xffff;
		const uint3 w0_1 = data.w0[localSlotIndex1];
		while (localSlotIndex2 != 0xffff) {
			const uint3 w0_2 = data.w0[localSlotIndex2];
			const uint32_t value2 = data.lookupTable[localSlotIndex2];
			const uint32_t slotIndex2 = value2 >> 16;
			localSlotIndex2 = value2 & 0xffff;

			const uint32_t xorWork = w0_1.x ^ w0_2.x;

			uint4 w0;
			w0.x = (xorWork & 0x0007ffff) | (bucketIndex << 19);
			w0.y = w0_1.y ^ w0_2.y;
			w0.z = w0_1.z ^ w0_2.z;
			w0.w = __byte_perm(slotIndex1, slotIndex2, 0x1054);

			const uint32_t outputBucketIndex = (xorWork >> 8) & 0x1fff;
			const uint32_t outputBucketSlotIndex = atomicAdd(context->bucketSizes3 + outputBucketIndex, 1);
			if (outputBucketIndex >= BucketLayout3::maxBucketSize)
				continue;

			const uint32_t outputRawSlotIndex = outputBucketIndex * BucketLayout3::maxBucketSize + outputBucketSlotIndex;
			*(uint4 *)(context->buckets3 + outputRawSlotIndex * 16) = w0;
		}
	}

	if ((mask == 0) &&
		(threadIndex == 0))
		context->bucketSizes1[bucketIndex] = 0;
}

__device__ __forceinline__ void extractIndices3(EquihashContext *context, uint32_t bucketIndex_3, uint32_t slotIndex_3, uint32_t &bucketIndex_2, uint32_t &slotIndex1_2, uint32_t &slotIndex2_2) {
	const uint32_t rawSlotIndex = bucketIndex_3 * BucketLayout3::maxBucketSize + slotIndex_3;
	uint4 w0 = *(uint4 *)(context->buckets3 + rawSlotIndex * 16);
	bucketIndex_2 = w0.x >> 19;
	slotIndex1_2 = w0.w & 0xffff;
	slotIndex2_2 = w0.w >> 16;
}

struct Round4Data {
	uint32_t ht[BucketLayout3::hastTableSize];
	uint2 w0[BucketLayout3::maxBucketSize2];
	uint32_t lookupTable[BucketLayout3::maxBucketSize2];
	uint32_t localBucketSize;
};

__global__ void round4(EquihashContext *context) {
	__shared__ Round4Data data;

	const uint32_t bucketIndex = blockIdx.x / BucketLayout3::masking;
	const uint32_t mask = blockIdx.x % BucketLayout3::masking;
	const uint32_t threadIndex = threadIdx.x;

	const uint32_t bucketSize = umin(context->bucketSizes3[bucketIndex], BucketLayout3::maxBucketSize);
	data.ht[threadIndex] = 0xffff;
	if (threadIndex == 0)
		data.localBucketSize = 0;

	__syncthreads();

	for (uint32_t slotIndex = threadIndex; slotIndex < bucketSize; slotIndex += blockDim.x) {
		const uint32_t rawSlotIndex = bucketIndex * BucketLayout3::maxBucketSize + slotIndex;
		uint4 w0 = *(uint4 *)(context->buckets3 + rawSlotIndex * 16);
		if (((w0.x >> 6) & 3) == mask) {
			const uint32_t localSlotIndex = atomicAdd(&data.localBucketSize, 1);
			if (localSlotIndex < BucketLayout3::maxBucketSize2) {
				const uint32_t collision = ((w0.x & 63) << 4) | (w0.y >> 28);
				const uint32_t prevLocalSlotIndex = atomicExch(data.ht + collision, localSlotIndex);
				data.w0[localSlotIndex] = make_uint2(w0.y, w0.z);
				data.lookupTable[localSlotIndex] = __byte_perm(slotIndex, prevLocalSlotIndex, 0x1054);
			}
		}
	}

	__syncthreads();

	const uint32_t localBucketSize = umin(data.localBucketSize, BucketLayout3::maxBucketSize2);
	for (uint32_t localSlotIndex1 = threadIndex; localSlotIndex1 < localBucketSize; localSlotIndex1 += blockDim.x) {
		const uint32_t value1 = data.lookupTable[localSlotIndex1];
		const uint32_t slotIndex1 = value1 >> 16;
		uint32_t localSlotIndex2 = value1 & 0xffff;
		const uint2 w0_1 = data.w0[localSlotIndex1];
		while (localSlotIndex2 != 0xffff) {
			const uint2 w0_2 = data.w0[localSlotIndex2];
			const uint32_t value2 = data.lookupTable[localSlotIndex2];
			const uint32_t slotIndex2 = value2 >> 16;
			localSlotIndex2 = value2 & 0xffff;

			uint4 w0;
			w0.x = w0_1.x ^ w0_2.x;
			w0.y = w0_1.y ^ w0_2.y;
			w0.z = bucketIndex;
			w0.w = __byte_perm(slotIndex1, slotIndex2, 0x1054);

			const uint32_t outputBucketIndex = (w0.x >> 15) & 0x1fff;
			const uint32_t outputBucketSlotIndex = atomicAdd(context->bucketSizes4 + outputBucketIndex, 1);
			if (outputBucketIndex >= BucketLayout4::maxBucketSize)
				continue;

			const uint32_t outputRawSlotIndex = outputBucketIndex * BucketLayout4::maxBucketSize + outputBucketSlotIndex;
			*(uint4 *)(context->buckets4 + outputRawSlotIndex * 16) = w0;
		}
	}
	if ((mask == 0) &&
		(threadIndex == 0)) {
		context->bucketSizes2[bucketIndex] = 0;
		if (bucketIndex == 0) {
			context->candidateCount = 0;
			context->solutionCount = 0;
		}
	}
}

__device__ __forceinline__ void extractIndices4(EquihashContext *context, uint32_t bucketIndex_4, uint32_t slotIndex_4, uint32_t &bucketIndex_3, uint32_t &slotIndex1_3, uint32_t &slotIndex2_3) {
	const uint32_t rawSlotIndex = bucketIndex_4 * BucketLayout4::maxBucketSize + slotIndex_4;
	uint4 w0 = *(uint4 *)(context->buckets4 + rawSlotIndex * 16);
	bucketIndex_3 = w0.z;
	slotIndex1_3 = w0.w & 0xffff;
	slotIndex2_3 = w0.w >> 16;
}

struct Round5_0Data {
	uint32_t ht[BucketLayout4::hastTableSize];
	uint32_t w0[BucketLayout4::maxBucketSize2];
	uint32_t lookupTable[BucketLayout4::maxBucketSize2];
	uint32_t localBucketSize;
};

__global__ void round5_0(EquihashContext *context) {
	__shared__ Round5_0Data data;

	const uint32_t bucketIndex = blockIdx.x / BucketLayout4::masking;
	const uint32_t mask = blockIdx.x % BucketLayout4::masking;
	const uint32_t threadIndex = threadIdx.x;

	const uint32_t bucketSize = umin(context->bucketSizes4[bucketIndex], BucketLayout4::maxBucketSize);
	data.ht[threadIndex] = 0xffff;
	if (threadIndex == 0)
		data.localBucketSize = 0;

	__syncthreads();

	for (uint32_t slotIndex = threadIndex; slotIndex < bucketSize; slotIndex += blockDim.x) {
		const uint32_t rawSlotIndex = bucketIndex * BucketLayout4::maxBucketSize + slotIndex;
		uint4 w0 = *(uint4 *)(context->buckets4 + rawSlotIndex * 16);
		if (((w0.x >> 13) & 3) == mask) {
			const uint32_t localSlotIndex = atomicAdd(&data.localBucketSize, 1);
			if (localSlotIndex < BucketLayout4::maxBucketSize2) {
				const uint32_t collision = (w0.x >> 3) & 1023;
				const uint32_t prevLocalSlotIndex = atomicExch(data.ht + collision, localSlotIndex);
				data.w0[localSlotIndex] = ((w0.x & 7) << 22) | (w0.y >> 10);
				data.lookupTable[localSlotIndex] = __byte_perm(slotIndex, prevLocalSlotIndex, 0x1054);
			}
		}
	}

	__syncthreads();

	const uint32_t localBucketSize = umin(data.localBucketSize, BucketLayout4::maxBucketSize2);
	for (uint32_t localSlotIndex1 = threadIndex; localSlotIndex1 < localBucketSize; localSlotIndex1 += blockDim.x) {
		const uint32_t value1 = data.lookupTable[localSlotIndex1];
		const uint32_t slotIndex1 = value1 >> 16;
		uint32_t localSlotIndex2 = value1 & 0xffff;
		const uint32_t w0_1 = data.w0[localSlotIndex1];
		while (localSlotIndex2 != 0xffff) {
			const uint32_t w0_2 = data.w0[localSlotIndex2];
			const uint32_t value2 = data.lookupTable[localSlotIndex2];
			const uint32_t slotIndex2 = value2 >> 16;
			localSlotIndex2 = value2 & 0xffff;
			if (w0_1 == w0_2) {
				if (w0_1 == 0)
					continue;
				const uint32_t candidateIndex = atomicAdd(&context->candidateCount, 1);
				if (candidateIndex < BucketLayout4::maxCandidateCount) {
					uint2 candidate;
					candidate.x = bucketIndex;
					candidate.y = __byte_perm(slotIndex1, slotIndex2, 0x1054);
					context->candidates[candidateIndex] = candidate;
				}
			}
		}
	}

	if ((mask == 0) &&
		(threadIndex == 0)) {
		context->bucketSizes3[bucketIndex] = 0;
	}
}

struct Round5_1Data {
	uint32_t stringIndices[32];
	uint32_t duplicate;
	uint32_t solutionIndex;
};

__global__ void round5_1(EquihashContext *context) {
	__shared__ Round5_1Data data;

	const uint32_t candidateIndex = blockIdx.x;
	const uint32_t candidateCount = umin(context->candidateCount, BucketLayout4::maxCandidateCount);
	if (candidateIndex >= candidateCount)
		return;
	const uint32_t threadIndex = threadIdx.x;
	const uint2 candidate = context->candidates[candidateIndex];
	const uint32_t bucketIndex_4 = candidate.x;
	const uint32_t slotIndex1_4 = candidate.y & 0xffff;
	const uint32_t slotIndex2_4 = candidate.y >> 16;
	uint32_t bucketIndex_3, slotIndex1_3, slotIndex2_3;
	extractIndices4(context, bucketIndex_4, threadIndex & 16 ? slotIndex2_4 : slotIndex1_4, bucketIndex_3, slotIndex1_3, slotIndex2_3);
	uint32_t bucketIndex_2, slotIndex1_2, slotIndex2_2;
	extractIndices3(context, bucketIndex_3, threadIndex & 8 ? slotIndex2_3 : slotIndex1_3, bucketIndex_2, slotIndex1_2, slotIndex2_2);
	uint32_t bucketIndex_1, slotIndex1_1, slotIndex2_1;
	extractIndices2(context, bucketIndex_2, threadIndex & 4 ? slotIndex2_2 : slotIndex1_2, bucketIndex_1, slotIndex1_1, slotIndex2_1);
	uint32_t stringIndex1, stringIndex2;
	extractIndices1(context, bucketIndex_1, threadIndex & 2 ? slotIndex2_1 : slotIndex1_1, stringIndex1, stringIndex2);
	uint32_t stringIndex = threadIndex & 1 ? stringIndex1 : stringIndex2;
	data.stringIndices[threadIndex] = stringIndex;

	if (threadIndex == 0)
		data.duplicate = 0;

	__syncthreads();

	for (uint32_t i = threadIndex + 1; i < 32; ++i) {
		if (stringIndex == data.stringIndices[i]) {
			atomicExch(&data.duplicate, 1);
		}
	}

	__syncthreads();

	if (data.duplicate)
		return;

	if (threadIndex == 0)
		data.solutionIndex = atomicAdd(&context->solutionCount, 1);

	__syncthreads();

	if (data.solutionIndex >= EquihashContext::maxSolutionCount)
		return;

	context->solutions[data.solutionIndex * 32 + threadIndex] = stringIndex;
}
