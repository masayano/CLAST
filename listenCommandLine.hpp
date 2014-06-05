#ifndef LISTEN_COMMAND_LINE_HPP_
#define LISTEN_COMMAND_LINE_HPP_

#include <string>
#include <vector>

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
		std::string& outputFile);

#endif
