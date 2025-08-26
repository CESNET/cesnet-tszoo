# Benchmarks

CESNET-TS-Zoo enables easy sharing and reuse of configuration files to support open science, reproducibility, and transparent comparison of time series modeling approaches.

We provide a collection of pre-defined configurations that serve as benchmarks, including use cases like network traffic forecasting and anomaly detection.

The library includes tools for both importing and exporting configurations as benchmarks. This allows researchers to cite a specific benchmark via its unique hash or to share their own approach as a configuration file.

To load and use a benchmark in your code, simply use the following snippet:

```python
from cesnet_tszoo.benchmarks import load_benchmark

benchmark = load_benchmark("<benchmark_hash>", "<path-to-datasets>")
dataset = benchmark.get_initialized_dataset()
```

!!! info "Note"
    More detailed tutorial how to use benchmarks is available [`here`][benchmark_tutorial]

## Available benchmarks

#### Network Traffic Forecasting Benchmarks

Network traffic forecasting plays a crucial role in network management and security. Therefore, we prepared several benchmarks for evaluation of network traffic forecasting methods for both management and security tasks. We split the `Network Traffic Forecasting Benchmarks` into these two groups:

- ["Univariate forecasting - Transmitted data size"][univariate_forecasting]: Benchmarks in this group are designed to support mostly used forecasting task for network management.
- ["Multivariate forecasting"][multivariate_forecasting]: Benchmarks in this group are designed to multivariate forecasting of network traffic features which is more often usable in network security for anomaly/outlier detection.

#### Network Device Type Classification Benchmarks

Network device type classification focuses on evaluating the performance of models for classifying types of network devices. The goal of this benchmark is to allow comparison of various classification algorithms and methods in the context of network devices. This task is valuable in environments where it is essential to quickly and efficiently identify devices in a network for monitoring, security, and traffic optimization purposes. Analyzing the benchmarks helps determine which methods are most suitable for deployment in real-world scenarios.

The network device type classification benchmarks are described in detail: [here][device_type_classification]


#### Anomaly Detection Benchmarks

This benchmarks are in process of making and they will be added soon.

#### Similarity Search Benchmarks

This benchmarks are in process of making and they will be added soon.

## Available dataset benchmarks from related works

For supporting reproducibility of approaches, the CESNET-TS-Zoo allows to share ts-zoo benchmarks with others using pull request from forked repository.

Each related work contains benchmarks and example of usage. Please follow authors instruction in example to ensure comparable results. Following benchmarks are already included in the ts-zoo:

| DOI  | Task | Benchmarks link |
|:-----------------|:-----------------:|:-----------------:|
| <https://doi.org/10.48550/arXiv.2503.17410> | Univariate forecasting  | [benchmarks][arxiv.org/abs/2503.17410]  |
