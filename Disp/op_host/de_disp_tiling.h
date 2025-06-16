
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DeDispTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(float, time_reso);
TILING_DATA_FIELD_DEF(float, down_time_rate);
TILING_DATA_FIELD_DEF(float, xTeam);
TILING_DATA_FIELD_DEF(float, y);
TILING_DATA_FIELD_DEF(float, freq1);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DeDisp, DeDispTilingData)
}
