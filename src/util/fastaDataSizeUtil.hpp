#ifndef CLAST_FASTA_DATA_SIZE_UTIL_HPP_
#define CLAST_FASTA_DATA_SIZE_UTIL_HPP_

#include <string>
#include <vector>

/* Sum of line lengths in FASTA files, same rule as the former calcTotalDbSize in
 * CHostSetting: any line that contains '>' is skipped (treated as header or non-data). */
long calcTotalDbSizeFromFastaPaths(const std::vector<std::string>& filePaths);

#endif
