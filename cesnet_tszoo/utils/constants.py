from cesnet_tszoo.utils.enums import PreprocessType

# Time formats

UNIX_TIME_FORMAT = "unix_time"
SHIFTED_UNIX_TIME_FORMAT = "shifted_unix_time"
DATETIME_TIME_FORMAT = "datetime"
ID_TIME_FORMAT = "id_time"

# Column names

ID_TIME_COLUMN_NAME = "id_time"
TIME_COLUMN_NAME = "time"

# Other

ROW_START = "start"
ROW_END = "end"
LOADING_WARNING_THRESHOLD = 20_000_000
ANNOTATIONS_DOWNLOAD_BUCKET = "https://liberouter.org/datazoo/download?bucket=annotations"
MANDATORY_PREPROCESSES_ORDER = set([PreprocessType.HANDLING_ANOMALIES.value, PreprocessType.FILLING_GAPS.value, PreprocessType.TRANSFORMING.value])
