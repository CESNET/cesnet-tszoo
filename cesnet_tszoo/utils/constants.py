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

# Transformers

MIN_MAX_SCALER = "min_max_scaler"
STANDARD_SCALER = "standard_scaler"
MAX_ABS_SCALER = "max_abs_scaler"
LOG_TRANSFORMER = "log_transformer"
ROBUST_SCALER = "robust_scaler"
POWER_TRANSFORMER = "power_transformer"
QUANTILE_TRANSFORMER = "quantile_transformer"
L2_NORMALIZER = "l2_normalizer"

# Anomaly handlers

Z_SCORE = "z-score"
INTERQUARTILE_RANGE = "interquartile_range"
