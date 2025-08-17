# Multivariate forecasting { #multivariate_forecasting }

We divided these benchmarks into two groups.

### Unique model for each time series

First group target on training model for each time series. Available benchmarks for training unique model per each time series are here:

| Benchmark hash | Dataset | Aggregation | Source | Original paper |
|:-----------------|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
[930f0b401065][930f0b401065] | CESNET-TimeSeries24 | 10 MINUTES | INSTITUTIONS | None |
[ca6999ea7e24][ca6999ea7e24] | CESNET-TimeSeries24 | 10 MINUTES | INSTITUTION_SUBNETS | None |
[7495b16f5fe6][7495b16f5fe6] | CESNET-TimeSeries24 | 10 MINUTES | IP_ADDRESSES_FULL | None |
[3687fb52c433][3687fb52c433] | CESNET-TimeSeries24 | 10 MINUTES | IP_ADDRESSES_SAMPLE | None |
[a6e56f99ab8a][a6e56f99ab8a] | CESNET-TimeSeries24 | 1 HOUR | INSTITUTION_SUBNETS | None |
[e44334732033][e44334732033] | CESNET-TimeSeries24 | 1 HOUR | INSTITUTIONS | None |
[18d04cab63e4][18d04cab63e4] | CESNET-TimeSeries24 | 1 HOUR | IP_ADDRESSES_FULL | None |
[b0ea46897cae][b0ea46897cae] | CESNET-TimeSeries24 | 1 HOUR | IP_ADDRESSES_SAMPLE | None |
[63e1f696e7c5][63e1f696e7c5] | CESNET-TimeSeries24 | 1 DAY | INSTITUTION_SUBNETS | None |
[71d17ad3550f][71d17ad3550f] | CESNET-TimeSeries24 | 1 DAY | INSTITUTIONS | None |
[15737f3fceec][15737f3fceec] | CESNET-TimeSeries24 | 1 DAY | IP_ADDRESSES_FULL | None |
[084f368f4c82][084f368f4c82] | CESNET-TimeSeries24 | 1 DAY | IP_ADDRESSES_SAMPLE | None |

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

Second group target on training one generic model which learns generic paterns in several time series and then it can forecast multiple other time series. Available benchmarks for training unique model per each time series are here:

| Benchmark hash | Dataset | Aggregation | Source | Original paper |
|:-----------------|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
[9ac2b87c9a7c][9ac2b87c9a7c] | CESNET-TimeSeries24 | 10 MINUTES | INSTITUTIONS | None |
[7cd4e41b05ec][7cd4e41b05ec] | CESNET-TimeSeries24 | 10 MINUTES | INSTITUTION_SUBNETS | None |
[50eb509e1e77][50eb509e1e77] | CESNET-TimeSeries24 | 10 MINUTES | IP_ADDRESSES_FULL | None |
[681a7fb90948][681a7fb90948] | CESNET-TimeSeries24 | 10 MINUTES | IP_ADDRESSES_SAMPLE | None |
[ab8183ea80af][ab8183ea80af] | CESNET-TimeSeries24 | 1 HOUR | INSTITUTION_SUBNETS | None |
[f9bd005c7efe][f9bd005c7efe] | CESNET-TimeSeries24 | 1 HOUR | INSTITUTIONS | None |
[88fd173619b2][88fd173619b2] | CESNET-TimeSeries24 | 1 HOUR | IP_ADDRESSES_FULL | None |
[4ae11863ee38][4ae11863ee38] | CESNET-TimeSeries24 | 1 HOUR | IP_ADDRESSES_SAMPLE | None |
[cdb79dbf54ea][cdb79dbf54ea] | CESNET-TimeSeries24 | 1 DAY | INSTITUTION_SUBNETS | None |
[c95d66b0baf5][c95d66b0baf5] | CESNET-TimeSeries24 | 1 DAY | INSTITUTIONS | None |
[16274e0b44af][16274e0b44af] | CESNET-TimeSeries24 | 1 DAY | IP_ADDRESSES_FULL | None |
[0197980a87c0][0197980a87c0] | CESNET-TimeSeries24 | 1 DAY | IP_ADDRESSES_SAMPLE | None |

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
dataset.apply_transformer(TransformerType.MIN_MAX_SCALER)

# (optional) Change sliding window setting
dataset.set_sliding_window(sliding_window_size=744, sliding_window_prediction_size=24, sliding_window_step=1, set_shared_size=744)

# (optional) Change batch sizes
dataset.set_batch_sizes(all_batch_size=32)

# or to update all at once which is usually faster
# dataset.update_dataset_config_and_initialize(default_values=0, sliding_window_size=744, sliding_window_prediction_size=24, sliding_window_step=1, set_shared_size=744, 
#                                              fill_missing_with=FillerType.MEAN_FILLER, transform_with=TransformerType.MIN_MAX_SCALER, all_batch_size=32)

# Process with your own defined model
model = Model()
model.fit(
    dataset.get_train_dataloader(), 
    dataset.get_val_dataloader(),
)

# Predict for time series which data are not in training
y_pred, y_true = model.predict(
    dataset.get_test_dataloader(), 
)
    
# Evaluate predictions, for example, with RMSE
rmse = mean_squared_error(y_true, y_pred)
print(f"RMSE: {rmse:.4f}")
```
