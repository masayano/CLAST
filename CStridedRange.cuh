#ifndef C_STRIDED_RANGE_CUH_
#define C_STRIDED_RANGE_CUH_

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

#include <thrust/device_vector.h>

template <class Iterator> class CStridedRange {
	int stride;
	Iterator first;
	Iterator last;
public:
	typedef typename thrust::iterator_difference<Iterator>::type difference_type;

	struct StrideFunctor : public thrust::unary_function<difference_type, difference_type> {
		int stride;

		StrideFunctor(int stride) : stride(stride) {}
		__device__ difference_type operator()(const difference_type& i) const { return stride * i; }
    };

	typedef class thrust::counting_iterator<difference_type>						CountingIterator;
	typedef class thrust::transform_iterator<StrideFunctor, CountingIterator>		TransformIterator;
	typedef class thrust::permutation_iterator<Iterator,TransformIterator>		PermutationIterator;
	typedef PermutationIterator iterator;

	CStridedRange(void) {}
	CStridedRange(Iterator first, Iterator last, int stride) : first(first), last(last), stride(stride) {}
	iterator begin(void) const {
		return PermutationIterator(first, TransformIterator(CountingIterator(0), StrideFunctor(stride)));
	}
	iterator end(void) const {
		return begin() + ((last - first) + (stride - 1)) / stride;
	}
};

#endif
