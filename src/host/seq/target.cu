#include "host/seq/target.cuh"

#include "util/utilAddSequence.cuh"

/***************************** class function ****************************/

/* called at CHostMapper::addTarget() */
void CHostSeqList_target::add(
		const CHostSetting& setting,
		const std::vector<CHostFASTA>& seq) {
	const int jointLength = setting.getLMerLength() - 1;
	size_t addLengthTotal = 0;
	for(std::vector<CHostFASTA>::const_iterator i = seq.begin(); i != seq.end(); ++i) {
		addLengthTotal += (*i).getSequence().size() + jointLength;
	}
	indexArray.reserve(indexArray.size() + addLengthTotal);
	IDArray   .reserve(IDArray   .size() + addLengthTotal);
	baseArray .reserve(baseArray .size() + addLengthTotal);

	for(std::vector<CHostFASTA>::const_iterator i = seq.begin(); i != seq.end(); ++i) {
		using namespace thrust;

		const std::string& FASTAseq = (*i).getSequence();
		const int seqLength = FASTAseq.size();

		labelArray   .push_back((*i).getLabel()   );
		startIdxArray.push_back((*i).getStartIdx());

		addSequence(
				seqLength,
				setting.getLMerLength(),
				FASTAseq,
				indexArray,
				IDArray,
				baseArray);

		lengthArray.push_back(seqLength);

		if(gateway.empty()) { gateway.push_back(0); }
		gateway.push_back(gateway.back() + seqLength + setting.getLMerLength() - 1);
	}
}

/* called at CHostResultHolder::addStartIdx() */
int CHostSeqList_target::getStartIdx(const int seqID) const {
	return startIdxArray.at(seqID);
}
