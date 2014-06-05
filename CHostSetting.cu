#include "CHostSetting.cuh"

#include "common.hpp"

#include "listenCommandLine.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

/**************************** validator *****************************/

void validator(
		const int lMerLength,
		const int strideLength,
		const int allowableGap,
		const std::vector<std::string>& targetFileArray,
		const std::vector<std::string>& queryFileArray,
		const std::string& outputFile) {
	if(lMerLength > MAX_L_MER_LENGTH) {
		std::cout << " error : too large lMerLength was inputted." << std::endl;
		abort();
	}
	if(allowableGap > MAX_ALLOWABLE_GAP) {
		std::cout << " error : too large allowableGap was inputted." << std::endl;
		abort();
	} else if(allowableGap < 2) {
		std::cout << " error : allowableGap must be larger than 1." << std::endl;
		abort();
	}
	if(targetFileArray.empty()) {
		std::cout << " error : no target file was inputted." << std::endl;
		abort();
	}
	if(queryFileArray.empty()) {
		std::cout << " error : no query file was inputted." << std::endl;
		abort();
	}
	if(outputFile.empty()) {
		std::cout << " error : no output file was inputted." << std::endl;
		abort();
	}
}

long calcTotalDbSize(const std::vector<std::string>& targetFileArray) {
	long totalDbSize = 0;
	for(int i = 0; i < targetFileArray.size(); ++i) {
		std::ifstream ifs(targetFileArray[i].c_str(), std::ios::binary);
		std::string buf;
		while(ifs && std::getline(ifs, buf)) {
			if(buf.find(">") == std::string::npos) {
				totalDbSize += buf.size();
			}
		}
	}
	return totalDbSize;
}

/**************************** class function *********************************/

CHostSetting::CHostSetting(const int argc, const char** argv) {
	/* load default setting */
	targetRAMSize     = DEFAULT_TARGET_RAM_SIZE;
	queryRAMSize      = DEFAULT_QUERY_RAM_SIZE;
	targetVRAMSize    = DEFAULT_TARGET_VRAM_SIZE;
	queryVRAMSize     = DEFAULT_QUERY_VRAM_SIZE;
	lMerLength        = DEFAULT_L_MER_LENGTH;
	strideLength      = DEFAULT_STRIDE_LENGTH;
	cutRepeat         = DEFAULT_CUT_REPEAT;
	allowableWidth    = DEFAULT_ALLOWABLE_WIDTH;
	allowableGap      = DEFAULT_ALLOWABLE_GAP;
	numberOfOutput    = DEFAULT_NUMBER_OF_OUTPUT;
	flgLocal          = DEFAULT_FLG_LOCAL;
	deviceID          = DEFAULT_DEVICE_ID;
	sleepTime         = DEFAULT_SLEEP_TIME;
	cutOff            = DEFAULT_CUT_OFF;

	/* listen user */
	listenCommandLine(
			argc,
			argv,
			targetRAMSize,
			queryRAMSize,
			targetVRAMSize,
			queryVRAMSize,
			lMerLength,
			strideLength,
			cutRepeat,
			allowableWidth,
			allowableGap,
			numberOfOutput,
			flgLocal,
			deviceID,
			sleepTime,
			cutOff,
			targetFileArray,
			queryFileArray,
			outputFile);
	validator(
			lMerLength,
			strideLength,
			allowableGap,
			targetFileArray,
			queryFileArray,
			outputFile);

	/* print setting */
	std::cout << " Search setting :" << std::endl;
	std::cout << "  RAM usage for target (MB): " << targetRAMSize     << std::endl;
	std::cout << "  RAM usage for query  (MB): " <<  queryRAMSize     << std::endl;
	std::cout << "  VRAM usage for target(MB): " << targetVRAMSize    << std::endl;
	std::cout << "  VRAM usage for query (MB): " << queryVRAMSize     << std::endl;
	std::cout << "  l-mer length             : " << lMerLength        << std::endl;
	std::cout << "  stride length            : " << strideLength      << std::endl;
	std::cout << "  cut repeat num           : " << cutRepeat         << std::endl;
	std::cout << "  allowable seed-seed width: " << allowableWidth    << std::endl;
	std::cout << "  allowable gap            : " << allowableGap      << std::endl;
	std::cout << "  number of output         : " << numberOfOutput    << std::endl;
	std::cout << "  cut off E-Value score    : " << cutOff            << std::endl;
	std::cout << "  alignment type           : ";
	if(flgLocal) { std::cout << "local"; } else { std::cout << "global"; }
	std::cout << std::endl;
	std::cout << "  device ID                : " << deviceID          << std::endl;
	std::cout << "  sleeping interval time   : " << sleepTime         << std::endl;
	std::cout << "  target file list         : " << std::endl;
	for(int i = 0; i < targetFileArray.size(); ++i) { std::cout << "   No." << i+1 << " " << targetFileArray[i] << std::endl; }
	std::cout << "  query file list          : " << std::endl;
	for(int i = 0; i < queryFileArray .size(); ++i) { std::cout << "   No." << i+1 << " " << queryFileArray[i]  << std::endl; }
	std::cout << "  output file name         : " << outputFile << std::endl;

	/* edit RAM size data */
	targetRAMSize  *= (1000 * 1000);
	queryRAMSize   *= (1000 * 1000);
	targetVRAMSize *= (1000 * 1000);
	queryVRAMSize  *= (1000 * 1000);

	/* create non user-editable option */
	totalDatabaseSize = static_cast<double>(calcTotalDbSize(targetFileArray));
	K      = GLOBAL_K;
	lambda = GLOBAL_LAMBDA;
}

int CHostSetting::getTargetRAMSize  (void) const { return targetRAMSize; }
int CHostSetting::getQueryRAMSize   (void) const { return queryRAMSize; }
int CHostSetting::getTargetVRAMSize (void) const { return targetVRAMSize; }
int CHostSetting::getQueryVRAMSize  (void) const { return queryVRAMSize; }
int CHostSetting::getLMerLength     (void) const { return lMerLength; }
int CHostSetting::getStrideLength   (void) const { return strideLength; }
int CHostSetting::getCutRepeat      (void) const { return cutRepeat; }
int CHostSetting::getAllowableWidth (void) const { return allowableWidth; }
int CHostSetting::getAllowableGap   (void) const { return allowableGap; }
int CHostSetting::getNumberOfOutput (void) const { return numberOfOutput; }
int CHostSetting::getFlgLocal       (void) const { return flgLocal; }
int CHostSetting::getDeviceID       (void) const { return deviceID; }
int CHostSetting::getSleepTime      (void) const { return sleepTime; }
double CHostSetting::getCutOff(void) const { return cutOff; }
const std::vector<std::string>& CHostSetting::getTargetFileArray(void) const { return targetFileArray; }
const std::vector<std::string>& CHostSetting::getQueryFileArray (void) const { return queryFileArray; }
const std::string& CHostSetting::getOutputFile(void) const { return outputFile; }
double CHostSetting::getTotalDatabaseSize(void) const { return totalDatabaseSize; }
double CHostSetting::getK                (void) const { return K; }
double CHostSetting::getLambda           (void) const { return lambda; }
