#include "CHostSeqList_query.cuh"

#include "utilAddSequence.cuh"
#include "utilReverseSeq.hpp"

#include <thrust/host_vector.h>

/***************************** class function ****************************/

/* called at CHostMapper::addQuery() */
void CHostSeqList_query::add(
		const CHostSetting& setting,
		const std::vector<CHostFASTA>& seq) {
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
