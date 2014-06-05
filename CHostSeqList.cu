#include "CHostSeqList.cuh"

#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

/***************************** class function ****************************/

/* called at CHostResultHolder::addLabel() */
const std::string& CHostSeqList::getLabel(const int seqID) const {
	return labelArray.at(seqID);
}

/* called at CDeviceSeqList::CDeviceSeqList() */
const thrust::device_vector<int>  CHostSeqList::getIndexArray(const int startID, const int endID) const {
	using namespace thrust;

	host_vector<int>::const_iterator startIter = indexArray.begin() + getGatewayIdx(startID);
	host_vector<int>::const_iterator endIter   = indexArray.begin() + getGatewayIdx(endID);
	device_vector<int> output(startIter, endIter);

	return output;
}

/* called at CDeviceSeqList::CDeviceSeqList() */
const thrust::device_vector<int>  CHostSeqList::getIDArray   (const int startID, const int endID) const {
	using namespace thrust;

	host_vector<int>::const_iterator startIter = IDArray.begin() + getGatewayIdx(startID);
	host_vector<int>::const_iterator endIter   = IDArray.begin() + getGatewayIdx(endID);
	device_vector<int> output(startIter, endIter);

	return output;
}

/* called at CDeviceSeqList::CDeviceSeqList() */
const thrust::device_vector<char> CHostSeqList::getBaseArray (const int startID, const int endID) const {
	using namespace thrust;

	host_vector<char>::const_iterator startIter = baseArray.begin() + getGatewayIdx(startID);
	host_vector<char>::const_iterator endIter   = baseArray.begin() + getGatewayIdx(endID);
	device_vector<char> output(startIter, endIter);

	return output;
}

/* called at CDeviceSeqList::CDeviceSeqList() */
const thrust::device_vector<int> CHostSeqList::getLengthArray(const int startID, const int endID) const {
	using namespace thrust;

	host_vector<int>::const_iterator startIter = lengthArray.begin() + startID;
	host_vector<int>::const_iterator endIter   = lengthArray.begin() + endID;
	device_vector<int> output(startIter, endIter);

	return output;
}

/* called at CDeviceSeqList::CDeviceSeqList() */
const thrust::device_vector<int> CHostSeqList::getGateway    (const int startID, const int endID) const {
	using namespace thrust;

	host_vector<int>::const_iterator startIter = gateway.begin() + startID;
	host_vector<int>::const_iterator endIter   = gateway.begin() + endID;
	device_vector<int> output(startIter, endIter);

	constant_iterator<int> cut(*startIter);
	thrust::transform(output.begin(), output.end(), cut, output.begin(), minus<int>());

	return output;
}

/* called at CHostSchedular::search() */
int CHostSeqList::getGatewayIdx (const int seqID) const {
	return gateway[seqID];
}

/* called at CHostSchedular::search() */
int CHostSeqList::getGatewaySize(void) const {
	return gateway.size();
}
