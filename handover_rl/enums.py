from enum import Enum, IntEnum

class SliceType(IntEnum):
    EMBB = 0
    URLLC = 1

class TrafficClass(str, Enum):
    PAYLOAD = "payload"
    CONTROL = "control"

class HandoverType(IntEnum):
    NO_HO = 0
    INTRA_DU_INTRA_CU = 1
    INTER_DU_INTRA_CU = 2
    INTER_CU = 3
    