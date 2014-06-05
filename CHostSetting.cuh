#ifndef C_HOST_SETTING_CUH_
#define C_HOST_SETTING_CUH_

#include <string>
#include <vector>

class CHostSetting {
	int targetRAMSize;
	int queryRAMSize;
	int targetVRAMSize;
	int queryVRAMSize;
	int lMerLength;
	int strideLength;
	int cutRepeat;
	int allowableWidth;
	int allowableGap;
	int numberOfOutput;
	int flgLocal;
	int deviceID;
	int sleepTime;
	double cutOff;
	std::vector<std::string> targetFileArray;
	std::vector<std::string> queryFileArray;
	std::string outputFile;
	double totalDatabaseSize;
	double K;
	double lambda;
public:
	CHostSetting(const int argc, const char** argv);
	int getTargetRAMSize  (void) const;
	int getQueryRAMSize   (void) const;
	int getTargetVRAMSize (void) const;
	int getQueryVRAMSize  (void) const;
	int getLMerLength     (void) const;
	int getStrideLength   (void) const;
	int getCutRepeat      (void) const;
	int getAllowableWidth (void) const;
	int getAllowableGap   (void) const;
	int getNumberOfOutput (void) const;
	int getFlgLocal       (void) const;
	int getDeviceID       (void) const;
	int getSleepTime      (void) const;
	double getCutOff(void) const;
	const std::vector<std::string>& getTargetFileArray(void) const;
	const std::vector<std::string>& getQueryFileArray (void) const;
	const std::string& getOutputFile(void) const;
	double getTotalDatabaseSize(void) const;
	double getK                (void) const;
	double getLambda           (void) const;
};

#endif
