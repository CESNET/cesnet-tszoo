# Univariate forecasting { #univariate_forecasting }

Benchmarks in this group are designed to support mostly used forecasting task for network management. We divided these benchmarks into two groups.

### Unique model for each time series

First group target on training model for each time series. Available benchmarks for training unique model per each time series are here:

| Benchmark hash | Dataset | Aggregation | Source | Original paper |
|:-----------------|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
[22c5a8e8ffd3][22c5a8e8ffd3] | CESNET-TimeSeries24 | 10 MINUTES | INSTITUTIONS | None |
[095f847ca755][095f847ca755] | CESNET-TimeSeries24 | 10 MINUTES | INSTITUTION_SUBNETS | None |
[c2970e89d824][c2970e89d824] | CESNET-TimeSeries24 | 10 MINUTES | IP_ADDRESSES_SAMPLE | None |
[ddb1f02dae43][ddb1f02dae43] | CESNET-TimeSeries24 | 10 MINUTES | IP_ADDRESSES_FULL | None |
[871f5972109e][871f5972109e] | CESNET-TimeSeries24 | 1 HOUR | INSTITUTIONS | None |
[080582bcd519][080582bcd519] | CESNET-TimeSeries24 | 1 HOUR | INSTITUTION_SUBNETS | None |
[f3fc14310e2e][f3fc14310e2e] | CESNET-TimeSeries24 | 1 HOUR | IP_ADDRESSES_SAMPLE | None |
[e268fa9957f2][e268fa9957f2] | CESNET-TimeSeries24 | 1 HOUR | IP_ADDRESSES_FULL | None |
[0d523e69c328][0d523e69c328] | CESNET-TimeSeries24 | 1 DAY | INSTITUTIONS | None |
[8e2a07fb3177][8e2a07fb3177] | CESNET-TimeSeries24 | 1 DAY | INSTITUTION_SUBNETS | None |
[b5e5ea044b81][b5e5ea044b81] | CESNET-TimeSeries24 | 1 DAY | IP_ADDRESSES_SAMPLE | None |
[d19ba386743f][d19ba386743f] | CESNET-TimeSeries24 | 1 DAY | IP_ADDRESSES_FULL | None |

We encourage users to change default value for missing values, filler, transformer, sliding window step,  and batch sizes. However, users may not change the rest of the arguments. Usage of these benchmarks are following:

```python

from cesnet_tszoo.benchmarks import load_benchmark
from cesnet_tszoo.utils.enums import FillerType, TransformerType
from sklearn.metrics import mean_squared_error

benchmark = load_benchmark("871f5972109e", "../")
dataset = benchmark.get_initialized_dataset()

# (optional) Set default value for missing data 
dataset.set_default_values(0)

# (optional) Set filler for filling missing data 
dataset.apply_filler(FillerType.MEAN_FILLER)

# (optional) Set transformer for data
dataset.apply_transformer(TransformerType.MIN_MAX_SCALER)

# (optional) Change sliding window setting
dataset.set_sliding_window(sliding_window_size=744, sliding_window_prediction_size=24, sliding_window_step=1, set_shared_size=744)

# (optional) Change batch sizes
dataset.set_batch_sizes(all_batch_size=32)

# Process with model per each time series individualy 
results = []
for ts_id in dataset.get_data_about_set(about='train')['ts_ids']:
    # Define your own class Model uses dataloaders for perform training and prediction
    model = Model()
    model.fit(
        dataset.get_train_dataloader(ts_id), 
        dataset.get_val_dataloader(ts_id),
    )
    y_pred, y_true = model.predict(
        dataset.get_test_dataloader(ts_id), 
    )
    
    # Evaluate predictions, for example, with RMSE
    rmse = mean_squared_error(y_true, y_pred)
    
    # Add individual result into all results
    results.append(rmse)

print(f"Mean RMSE: {np.mean(rmse):.4f}")
print(f"Std RMSE: {np.std(rmse):.4f}")
```

### Generic model for multiple time series

Second group target on training one generic model which learns generic paterns in several time series and then it can forecast multiple other time series.  Available benchmarks for training unique model per each time series are here:

| Benchmark hash | Dataset | Aggregation | Source | Original paper |
|:-----------------|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
[7706f1087922][7706f1087922] | CESNET-TimeSeries24 | 10 MINUTES | INSTITUTIONS | None |
[a642915953ad][a642915953ad] | CESNET-TimeSeries24 | 10 MINUTES | INSTITUTION_SUBNETS | None |
[e3de1fc0a44e][e3de1fc0a44e] | CESNET-TimeSeries24 | 10 MINUTES | IP_ADDRESSES_SAMPLE | None |
[8b03d0d508ce][8b03d0d508ce] | CESNET-TimeSeries24 | 10 MINUTES | IP_ADDRESSES_FULL | None |
[09de83e89e42][09de83e89e42] | CESNET-TimeSeries24 | 1 HOUR | INSTITUTIONS | None |
[73a9add2c4af][73a9add2c4af] | CESNET-TimeSeries24 | 1 HOUR | INSTITUTION_SUBNETS | None |
[6249383544ef][6249383544ef] | CESNET-TimeSeries24 | 1 HOUR | IP_ADDRESSES_SAMPLE | None |
[b8098753b97b][b8098753b97b] | CESNET-TimeSeries24 | 1 HOUR | IP_ADDRESSES_FULL | None |
[ef632e70c252][ef632e70c252] | CESNET-TimeSeries24 | 1 DAY | INSTITUTIONS | None |
[ce63551ffaab][ce63551ffaab] | CESNET-TimeSeries24 | 1 DAY | INSTITUTION_SUBNETS | None |
[9f7047902d66][9f7047902d66] | CESNET-TimeSeries24 | 1 DAY | IP_ADDRESSES_SAMPLE | None |
[570b215d790d][570b215d790d] | CESNET-TimeSeries24 | 1 DAY | IP_ADDRESSES_FULL | None |

We encourage users to change default value for missing values, filler, transformer, sliding window step,  and batch sizes. However, users may not change the rest of the arguments. Usage of these benchmarks are following:

```python
from cesnet_tszoo.benchmarks import load_benchmark
from cesnet_tszoo.utils.enums import FillerType, TransformerType
from sklearn.metrics import mean_squared_error

benchmark = load_benchmark("09de83e89e42", "../")
dataset = benchmark.get_initialized_dataset()

# (optional) Set default value for missing data 
dataset.set_default_values(0)

# (optional) Set filler for filling missing data 
dataset.apply_filler(FillerType.MEAN_FILLER)

# (optional) Set transformer for data
dataset.apply_transformer(TransformerType.MIN_MAX_SCALER, create_transformer_per_time_series=False)

# (optional) Change sliding window setting
dataset.set_sliding_window(sliding_window_size=744, sliding_window_prediction_size=24, sliding_window_step=1, set_shared_size=744)

# (optional) Change batch sizes
dataset.set_batch_sizes(all_batch_size=32)

# Process with your own defined model
model = Model()
model.fit(
    dataset.get_train_dataloader(), 
    dataset.get_val_dataloader(),
)

# Predict for time series which data are not in training
y_pred, y_true = model.predict(
    dataset.get_test_other_dataloader(), 
)
    
# Evaluate predictions, for example, with RMSE
rmse = mean_squared_error(y_true, y_pred)
print(f"RMSE: {rmse::4f}")
```
