#include "host/seq/query.cuh"

#include "util/utilAddSequence.cuh"
#include "util/utilReverseSeq.hpp"

#include <thrust/host_vector.h>

/***************************** class function ****************************/

/* called at CHostMapper::addQuery() */
void CHostSeqList_query::add(
		const CHostSetting& setting,
		const std::vector<CHostFASTA>& seq) {
	const int jointLength = setting.getLMerLength() - 1;
	size_t addLengthTotal = 0;
	for(std::vector<CHostFASTA>::const_iterator i = seq.begin(); i != seq.end(); ++i) {
		addLengthTotal += (*i).getSequence().size() + jointLength;
	}
	addLengthTotal *= 2; // "+" and "-" strands
	indexArray.reserve(indexArray.size() + addLengthTotal);
	IDArray   .reserve(IDArray   .size() + addLengthTotal);
	baseArray .reserve(baseArray .size() + addLengthTotal);

	for(std::vector<CHostFASTA>::const_iterator i = seq.begin(); i != seq.end(); ++i) {
		using namespace thrust;

		const std::string& FASTAseq = (*i).getSequence();
		const int seqLength = FASTAseq.size();

		/* "+" strand */ {
			labelArray.push_back((*i).getLabel());

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

		/* "-" strand */ {
			labelArray.push_back((*i).getLabel());

			addSequence(
					seqLength,
					setting.getLMerLength(),
					compSeq(FASTAseq),
					indexArray,
					IDArray,
					baseArray);

			lengthArray.push_back(seqLength);

			if(gateway.empty()) { gateway.push_back(0); }
			gateway.push_back(gateway.back() + seqLength + setting.getLMerLength() - 1);
		}
	}
}
