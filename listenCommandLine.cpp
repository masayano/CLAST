#include "listenCommandLine.hpp"

#include <iostream>

#include <boost/lexical_cast.hpp>

enum STATE_LIST {
		START,
		T_RAM, Q_RAM, T_VRAM, Q_VRAM, L_MER, STRIDE, REPEAT, WIDTH, GAP, NUM_OUT,
		CUT_OFF, FLG_LOCAL, DEVICE, SLEEP,
		T_FILE, Q_FILE, O_FILE};

void listenCommandLine_start(
		const std::string& arg,
		STATE_LIST& state) {
	if(arg == "-tRAM") { state = T_RAM; }
	else if(arg == "-qRAM")     { state = Q_RAM; }
	else if(arg == "-tVRAM")    { state = T_VRAM; }
	else if(arg == "-qVRAM")    { state = Q_VRAM; }
	else if(arg == "-lMer")     { state = L_MER; }
	else if(arg == "-stride")   { state = STRIDE; }
	else if(arg == "-repeat")   { state = REPEAT; }
	else if(arg == "-width")    { state = WIDTH; }
	else if(arg == "-gap")      { state = GAP; }
	else if(arg == "-numOut")   { state = NUM_OUT; }
	else if(arg == "-cutOff")   { state = CUT_OFF; }
	else if(arg == "-local")    { state = FLG_LOCAL; }
	else if(arg == "-device")   { state = DEVICE; }
	else if(arg == "-sleep")    { state = SLEEP; }
	else if(arg == "-t")        { state = T_FILE; }
	else if(arg == "-q")        { state = Q_FILE; }
	else if(arg == "-o")        { state = O_FILE; }
	else {
		std::cout << "  error : Odd parameter was inputted. Check the command line parameter." << std::endl;
		abort();
	}
}

void listenCommandLine_tRAM(
		const std::string& arg,
		int& targetRAMSize,
		STATE_LIST& state) {
	try {
		targetRAMSize = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-tRAM\"." << std::endl;
		abort();
	}
}

void listenCommandLine_qRAM(
		const std::string& arg,
		int& queryRAMSize,
		STATE_LIST& state) {
	try {
		queryRAMSize = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-qRAM\"." << std::endl;
		abort();
	}
}

void listenCommandLine_tVRAM(
		const std::string& arg,
		int& targetVRAMSize,
		STATE_LIST& state) {
	try {
		targetVRAMSize = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-tVRAM\"." << std::endl;
		abort();
	}
}

void listenCommandLine_qVRAM(
		const std::string& arg,
		int& queryVRAMSize,
		STATE_LIST& state) {
	try {
		queryVRAMSize = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-qVRAM\"." << std::endl;
		abort();
	}
}

void listenCommandLine_lMer(
		const std::string& arg,
		int& lMerLength,
		STATE_LIST& state) {
	try {
		lMerLength = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-lMer\"." << std::endl;
		abort();
	}
}

void listenCommandLine_stride(
		const std::string& arg,
		int& strideLength,
		STATE_LIST& state) {
	try {
		strideLength = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-stride\"." << std::endl;
		abort();
	}
}

void listenCommandLine_repeat(
		const std::string& arg,
		int& cutRepeat,
		STATE_LIST& state) {
	try {
		cutRepeat = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-repeat\"." << std::endl;
		abort();
	}
}

void listenCommandLine_width(
		const std::string& arg,
		int& allowableWidth,
		STATE_LIST& state) {
	try {
		allowableWidth = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-width\"." << std::endl;
		abort();
	}
}

void listenCommandLine_gap(
		const std::string& arg,
		int& allowableGap,
		STATE_LIST& state) {
	try {
		allowableGap = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-gap\"." << std::endl;
		abort();
	}
}

void listenCommandLine_numOut(
		const std::string& arg,
		int& numberOfOutput,
		STATE_LIST& state) {
	try {
		numberOfOutput = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-numOut\"." << std::endl;
		abort();
	}
}

void listenCommandLine_flgLocal(
		const std::string& arg,
		int& flgLocal,
		STATE_LIST& state) {
	try {
		flgLocal = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-local\"." << std::endl;
		abort();
	}
}

void listenCommandLine_device(
		const std::string& arg,
		int& deviceID,
		STATE_LIST& state) {
	try {
		deviceID = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-device\"." << std::endl;
		abort();
	}
}

void listenCommandLine_sleep(
		const std::string& arg,
		int& sleepTime,
		STATE_LIST& state) {
	try {
		sleepTime = boost::lexical_cast<int>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-sleep\"." << std::endl;
		abort();
	}
}

void listenCommandLine_cutOff(
		const std::string& arg,
		double& cutOff,
		STATE_LIST& state) {
	try {
		cutOff = boost::lexical_cast<double>(arg);
		state = START;
	} catch (boost::bad_lexical_cast) {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-cutOff\"." << std::endl;
		abort();
	}
}

void listenCommandLine_tFile(
		const std::string& arg,
		std::vector<std::string>& targetFileArray,
		STATE_LIST& state) {
	if(arg == "-tRAM") { state = T_RAM; }
	else if(arg == "-qRAM")     { state = Q_RAM; }
	else if(arg == "-tVRAM")    { state = T_VRAM; }
	else if(arg == "-qVRAM")    { state = Q_VRAM; }
	else if(arg == "-lMer")     { state = L_MER; }
	else if(arg == "-stride")   { state = STRIDE; }
	else if(arg == "-repeat")   { state = REPEAT; }
	else if(arg == "-width")    { state = WIDTH; }
	else if(arg == "-gap")      { state = GAP; }
	else if(arg == "-numOut")   { state = NUM_OUT; }
	else if(arg == "-cutOff")   { state = CUT_OFF; }
	else if(arg == "-local")    { state = FLG_LOCAL; }
	else if(arg == "-device")   { state = DEVICE; }
	else if(arg == "-sleep")    { state = SLEEP; }
	else if(arg == "-t") {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-t\"." << std::endl;
		abort();
	} else if(arg == "-q")      { state = Q_FILE; }
	else if(arg == "-o")        { state = O_FILE; }
	else { targetFileArray.push_back(arg); }
}

void listenCommandLine_qFile(
		const std::string& arg,
		std::vector<std::string>& queryFileArray,
		STATE_LIST& state) {
	if(arg == "-tRAM") { state = T_RAM; }
	else if(arg == "-qRAM")     { state = Q_RAM; }
	else if(arg == "-tVRAM")    { state = T_VRAM; }
	else if(arg == "-qVRAM")    { state = Q_VRAM; }
	else if(arg == "-lMer")     { state = L_MER; }
	else if(arg == "-stride")   { state = STRIDE; }
	else if(arg == "-repeat")   { state = REPEAT; }
	else if(arg == "-width")    { state = WIDTH; }
	else if(arg == "-gap")      { state = GAP; }
	else if(arg == "-numOut")   { state = NUM_OUT; }
	else if(arg == "-cutOff")   { state = CUT_OFF; }
	else if(arg == "-local")    { state = FLG_LOCAL; }
	else if(arg == "-device")   { state = DEVICE; }
	else if(arg == "-sleep")    { state = SLEEP; }
	else if(arg == "-t")        { state = T_FILE; }
	else if(arg == "-q") {
		std::cout << "  error : Odd parameter was inputted. Check the after of \"-q\"." << std::endl;
		abort();
	} else if(arg == "-o")      { state = O_FILE; }
	else { queryFileArray.push_back(arg); }
}

void listenCommandLine_oFile(
		const std::string& arg,
		std::string& outputFile,
		STATE_LIST& state) {
	outputFile = arg;
	state = START;
}

void listenCommandLine(
		const int argc,
		const char** argv,
		int& targetRAMSize,
		int& queryRAMSize,
		int& targetVRAMSize,
		int& queryVRAMSize,
		int& lMerLength,
		int& strideLength,
		int& cutRepeat,
		int& allowableWidth,
		int& allowableGap,
		int& numberOfOutput,
		int& flgLocal,
		int& deviceID,
		int& sleepTime,
		double& cutOff,
		std::vector<std::string>& targetFileArray,
		std::vector<std::string>& queryFileArray,
		std::string& outputFile) {
	STATE_LIST state = START;
	for(int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		switch(state) {
			case START    : listenCommandLine_start   (arg, state); break;
			case T_RAM    : listenCommandLine_tRAM    (arg, targetRAMSize,   state); break;
			case Q_RAM    : listenCommandLine_qRAM    (arg, queryRAMSize,    state); break;
			case T_VRAM   : listenCommandLine_tVRAM   (arg, targetVRAMSize,  state); break;
			case Q_VRAM   : listenCommandLine_qVRAM   (arg, queryVRAMSize,   state); break;
			case L_MER    : listenCommandLine_lMer    (arg, lMerLength,      state); break;
			case STRIDE   : listenCommandLine_stride  (arg, strideLength,    state); break;
			case REPEAT   : listenCommandLine_repeat  (arg, cutRepeat,       state); break;
			case WIDTH    : listenCommandLine_width   (arg, allowableWidth,  state); break;
			case GAP      : listenCommandLine_gap     (arg, allowableGap,    state); break;
			case NUM_OUT  : listenCommandLine_numOut  (arg, numberOfOutput,  state); break;
			case CUT_OFF  : listenCommandLine_cutOff  (arg, cutOff,          state); break;
			case FLG_LOCAL: listenCommandLine_flgLocal(arg, flgLocal,        state); break;
			case DEVICE   : listenCommandLine_device  (arg, deviceID,        state); break;
			case SLEEP    : listenCommandLine_sleep   (arg, sleepTime,       state); break;
			case T_FILE   : listenCommandLine_tFile   (arg, targetFileArray, state); break;
			case Q_FILE   : listenCommandLine_qFile   (arg, queryFileArray,  state); break;
			case O_FILE   : listenCommandLine_oFile   (arg, outputFile,      state); break;
		}
	}
}
