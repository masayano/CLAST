#include "utilReverseSeq.hpp"

char compBase(const char base) {
	switch(base) {
		case 'A' : return 'T';
		case 'T' : return 'A';
		case 'G' : return 'C';
		case 'C' : return 'G';
		default : return 'N';
	}
}

std::string compSeq(const std::string& seq) {
	std::string comp;
	for(std::string::const_reverse_iterator it = seq.rbegin(); it != seq.rend(); ++it) {
		comp += compBase(*it);
	}
	return comp;
}
