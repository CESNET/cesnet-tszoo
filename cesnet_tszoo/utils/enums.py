from enum import Enum


class AgreggationType(Enum):
    """Possible aggregations for database. """

    AGG_1_DAY = "1_day"
    """1 day aggregation for source type. """

    AGG_1_HOUR = "1_hour"
    """1 hour aggregation for source type. """

    AGG_10_MINUTES = "10_minutes"
    """10 minutes aggregation for source type. """

    AGG_1_MINUTE = "1_minute"
    """1 minute aggregation for source type. """

    @staticmethod
    def _to_str_with_agg(aggregation_type):
        """For paths. """

        return f"agg_{aggregation_type.value}"

    @staticmethod
    def _to_str_without_number(aggregation_type) -> str:
        """For paths. """

        if aggregation_type == AgreggationType.AGG_10_MINUTES:
            return "minutes"
        elif aggregation_type == AgreggationType.AGG_1_HOUR:
            return "hour"
        elif aggregation_type == AgreggationType.AGG_1_DAY:
            return "day"
        elif aggregation_type == AgreggationType.AGG_1_MINUTE:
            return "minute"
        else:
            raise NotImplementedError()


class SourceType(Enum):
    """Possible source types for database. """

    IP_ADDRESSES_FULL = "ip_addresses_full"
    """Traffic of ip addresses of specific devices. """

    IP_ADDRESSES_SAMPLE = "ip_addresses_sample"
    """Traffic of subset from `ip_addresses_full`. """

    INSTITUTION_SUBNETS = "institution_subnets"
    """Traffic of subnets in institutions`. """

    INSTITUTIONS = "institutions"
    """Traffic Institutions of CESNET3 network. """

    CESNET2 = "CESNET2"
    """Traffic of CESNET2 network. """


class FillerType(Enum):
    """Built-in filler types. """

    MEAN_FILLER = "mean_filler"
    """Represents filler [`MeanFiller`][cesnet_tszoo.utils.filler.MeanFiller]. Equivalent to literal `mean_filler`. """

    FORWARD_FILLER = "forward_filler"
    """Represents filler [`ForwardFiller`][cesnet_tszoo.utils.filler.ForwardFiller]. Equivalent to literal `forward_filler`. """

    LINEAR_INTERPOLATION_FILLER = "linear_interpolation_filler"
    """Represents filler [`LinearInterpolationFiller`][cesnet_tszoo.utils.filler.LinearInterpolationFiller]. Equivalent to literal `linear_interpolation_filler`. """


class TransformerType(Enum):
    """Built-in transformer types. """

    MIN_MAX_SCALER = "min_max_scaler"
    """Represents transformer [`MinMaxScaler`][cesnet_tszoo.utils.transformer.MinMaxScaler]. Equivalent to literal `min_max_scaler`. """

    STANDARD_SCALER = "standard_scaler"
    """Represents transformer [`StandardScaler`][cesnet_tszoo.utils.transformer.StandardScaler]. Equivalent to literal `standard_scaler`. """

    MAX_ABS_SCALER = "max_abs_scaler"
    """Represents transformer [`MaxAbsScaler`][cesnet_tszoo.utils.transformer.MaxAbsScaler]. Equivalent to literal `max_abs_scaler`. """

    LOG_TRANSFORMER = "log_transformer"
    """Represents transformer [`LogTransformer`][cesnet_tszoo.utils.transformer.LogTransformer]. Equivalent to literal `log_transformer`. """

    L2_NORMALIZER = "l2_normalizer"
    """Represents transformer [`L2Normalizer`][cesnet_tszoo.utils.transformer.L2Normalizer]. Equivalent to literal `l2_normalizer`. """

    ROBUST_SCALER = "robust_scaler"
    """Represents transformer [`RobustScaler`][cesnet_tszoo.utils.transformer.LogTransformer]. Equivalent to literal `robust_scaler`. """

    POWER_TRANSFORMER = "power_transformer"
    """Represents transformer [`PowerTransformer`][cesnet_tszoo.utils.transformer.PowerTransformer]. Equivalent to literal `power_transformer`. """

    QUANTILE_TRANSFORMER = "quantile_transformer"
    """Represents transformer [`QuantileTransformer`][cesnet_tszoo.utils.transformer.QuantileTransformer]. Equivalent to literal `quantile_transformer`. """


class TimeFormat(Enum):
    """Different supported time formats. """

    ID_TIME = "id_time"
    """Time as indices, starting from 0. """

    DATETIME = "datetime"
    """Time as [`datetime`](https://docs.python.org/3/library/datetime.html) object. """

    UNIX_TIME = "unix_time"
    """Time in unix time format. """

    SHIFTED_UNIX_TIME = "shifted_unix_time"
    """Unix time format but offsetted so it starts from 0. """


class SplitType(Enum):
    """Different split variants. """

    TRAIN = "train"
    """Represents training set of dataset. """

    VAL = "val"
    """Represents validation set of dataset. """

    TEST = "test"
    """Represents test set of dataset. """

    ALL = "all"
    """Represents train/val/test sets as one or just everything. """


class AnnotationType(Enum):
    """Categories of Annotations. """

    ID_TIME = "id_time"
    """Represents annotations for time. """

    TS_ID = "ts_id"
    """Represents annotations for time series. """

    BOTH = "both"
    """Represents annotations for time in time series. """


class DataloaderOrder(Enum):
    """Order for loading data with PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). """

    RANDOM = "random"
    """Loaded data will be randomly selected. """

    SEQUENTIAL = "sequential"
    """Loaded data will be in the selected order. """


class DatasetType(Enum):
    """Types of datasets. """

    TIME_BASED = "time_based"
    """This type of dataset is defined by train/val/test time periods and one time series set. """

    SERIES_BASED = "series_based"
    """This type of dataset is defined by train/val/test time series sets and one time period set. """

    DISJOINT_TIME_BASED = "disjoint_time_based"
    """This type of dataset is defined by train/val/test time series sets and their respective time period sets. """


class ScalerType(Enum):
    """Obsolete, dont use. Only for backward compatibility. """

    MIN_MAX_SCALER = "min_max_scaler"
    """Represents transformer [`MinMaxScaler`][cesnet_tszoo.utils.transformer.MinMaxScaler]. Equivalent to literal `min_max_scaler`. """

    STANDARD_SCALER = "standard_scaler"
    """Represents transformer [`StandardScaler`][cesnet_tszoo.utils.transformer.StandardScaler]. Equivalent to literal `standard_scaler`. """

    MAX_ABS_SCALER = "max_abs_scaler"
    """Represents transformer [`MaxAbsScaler`][cesnet_tszoo.utils.transformer.MaxAbsScaler]. Equivalent to literal `max_abs_scaler`. """

    LOG_TRANSFORMER = "log_transformer"
    """Represents transformer [`LogTransformer`][cesnet_tszoo.utils.transformer.LogTransformer]. Equivalent to literal `log_transformer`. """

    L2_NORMALIZER = "l2_normalizer"
    """Represents transformer [`L2Normalizer`][cesnet_tszoo.utils.transformer.L2Normalizer]. Equivalent to literal `l2_normalizer`. """

    ROBUST_SCALER = "robust_scaler"
    """Represents transformer [`RobustScaler`][cesnet_tszoo.utils.transformer.LogTransformer]. Equivalent to literal `robust_scaler`. """

    POWER_TRANSFORMER = "power_transformer"
    """Represents transformer [`PowerTransformer`][cesnet_tszoo.utils.transformer.PowerTransformer]. Equivalent to literal `power_transformer`. """

    QUANTILE_TRANSFORMER = "quantile_transformer"
    """Represents transformer [`QuantileTransformer`][cesnet_tszoo.utils.transformer.QuantileTransformer]. Equivalent to literal `quantile_transformer`. """
