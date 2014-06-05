#ifndef COMMON_HPP_
#define COMMON_HPP_

/* If you want to run in test or time-attack mode,
 * only you have to do is to comment-in this definition.
 *
 */
//#define MODE_TEST
//#define TIME_ATTACK

/* default setting (config.txt) */
const int DEFAULT_TARGET_RAM_SIZE    = 64;
const int DEFAULT_QUERY_RAM_SIZE     = 64;
const int DEFAULT_TARGET_VRAM_SIZE   = 64;
const int DEFAULT_QUERY_VRAM_SIZE    = 2;
const int DEFAULT_L_MER_LENGTH       = 15;
const int DEFAULT_STRIDE_LENGTH      = 5;
const int DEFAULT_CUT_REPEAT         = 20;
const int DEFAULT_ALLOWABLE_WIDTH    = 100;
const int DEFAULT_ALLOWABLE_GAP      = 8;
const int DEFAULT_NUMBER_OF_OUTPUT   = -1;
const int DEFAULT_FLG_LOCAL          = 0;
const int DEFAULT_DEVICE_ID          = 0;
const int DEFAULT_SLEEP_TIME         = 0;
const double DEFAULT_CUT_OFF = 10;

/* option range setting (config.txt) */
const int MAX_L_MER_LENGTH  = 31;
const int MAX_ALLOWABLE_GAP = 16;

/* allignment size setting */
const int MAX_ALIGNMENT_WIDTH = 33;    // == 2 * (MAX_ALLIGNMENT_GAP) + 1
const int MARGIN = 1;

/* allignment point setting */
const char BAD_AREA_CHAR = 'X';
const int BAD_AREA_POINT = -1000;
const int MATCH_POINT    = 1;
const int MISMATCH_POINT = -3;
const int GAP_OPEN_POINT = -5;
const int GAP_POINT      = -2;
const double GLOBAL_K      = 0.71;
const double GLOBAL_LAMBDA = 1.37;

#endif /* COMMON_HPP_ */
