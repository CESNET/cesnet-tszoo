# Network Device Type Classification { #device_type_classification }

Network device type classification focuses on evaluating the performance of models for classifying types of network devices. The goal of this benchmark is to allow comparison of various classification algorithms and methods in the context of network devices. This task is valuable in environments where it is essential to quickly and efficiently identify devices in a network for monitoring, security, and traffic optimization purposes. Analyzing the benchmarks helps determine which methods are most suitable for deployment in real-world scenarios.

Available benchmarks for training unique model per each time series are here:

| Benchmark hash | Dataset | Aggregation | Source | Original paper |
|:-----------------|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
[69270dcc1819][69270dcc1819] | CESNET-TimeSeries24 | 10 MINUTES | IP_ADDRESSES_FULL | None |
[941261e8c367][941261e8c367] | CESNET-TimeSeries24 | 1 HOUR | IP_ADDRESSES_FULL | None |
[bf0aec939afe][bf0aec939afe] | CESNET-TimeSeries24 | 1 DAY | IP_ADDRESSES_FULL | None |

We encourage users to change default value for missing values, filler, scaler, sliding window step,  and batch sizes. However, users may not change the rest of the arguments. Usage of these benchmarks are following:

```python
from cesnet_tszoo.benchmarks import load_benchmark
from cesnet_tszoo.utils.enums import AnnotationType, FillerType, ScalerType, SplitType
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

benchmark = load_benchmark("bf0aec939afe", "../")
dataset = benchmark.get_initialized_dataset()

# Get annotations
annotations = benchmark.get_annotations(on=AnnotationType.TS_ID)

# Prepare annotations
encoder = LabelEncoder()
annotations['group_encoded'] = encoder.fit_transform(annotations['group'])

train_annotations = annotations[annotations['id_ip'].isin(dataset.get_data_about_set(about=SplitType.TRAIN)['ts_ids'])]
train_target = train_annotations['group_encoded'].to_numpy()

test_annotations = annotations[annotations['id_ip'].isin(dataset.get_data_about_set(about=SplitType.TEST)['test_ts_ids'])]
test_target = test_annotations['group_encoded'].to_numpy()

# (optional) Set default value for missing data 
dataset.set_default_values(0)

# (optional) Set filler for filling missing data 
dataset.apply_filler(FillerType.MEAN_FILLER)

# (optional) Set scaler for data
dataset.apply_scaler(ScalerType.MIN_MAX_SCALER, create_scaler_per_time_series=False)

# (optional) Change sliding window setting
dataset.set_sliding_window(sliding_window_size=744, sliding_window_prediction_size=24, sliding_window_step=1, set_shared_size=744)

# (optional) Change batch sizes
dataset.set_batch_sizes(all_batch_size=32)

# Process with your own defined model
model = Model()
model.fit(
    dataset.get_train_dataloader(), 
    dataset.get_val_dataloader(),
    train_target,
)

# Predict for time series which data are not in training
y_pred = model.predict(
    dataset.get_test_other_dataloader()
)

# Evaluate predictions, for example, with RMSE
accuracy = accuracy_score(test_target, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```
