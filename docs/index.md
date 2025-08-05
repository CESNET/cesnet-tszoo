# CESNET TSZoo

This is documentation of the [CESNET TSZoo](https://github.com/CESNET/cesnet-tszoo) project. 

The goal of `cesnet-tszoo` project is to provide time series datasets with useful tools for preprocessing and reproducibility. Such as:

- API for downloading, configuring and loading CESNET-TimeSeries24, CESNET-AGG23 datasets. Each with various sources and aggregations. Check [dataset overview][overview-of-datasets] page for details about datasets.
- Example of configuration options:
    - Data can be split into train/val/test sets. Split can be done by time series, check [`SeriesBasedCesnetDataset`][cesnet_tszoo.datasets.series_based_cesnet_dataset.SeriesBasedCesnetDataset], or by time periods, check [`TimeBasedCesnetDataset`][cesnet_tszoo.datasets.time_based_cesnet_dataset.TimeBasedCesnetDataset].
    - Transforming of data with built-in transformers or with custom transformers. Check [`transformers`][cesnet_tszoo.utils.transformer] for details.
    - Handling missing values built-in fillers or with custom fillers. Check [`fillers`][cesnet_tszoo.utils.filler] for details.
- Creation and import of benchmarks, for easy reproducibility of experiments.
- Creation and import of annotations. Can create annotations for specific time series, specific time or specific time in specific time series.