# Comparative Analysis of Deep Learning Models for Real-World ISP Network Traffic Forecasting { #arxiv.org/abs/2503.17410 }

These configs were used in the paper ["Koumar, J., Smoleň, T., Jeřábek, K. and Čejka, T., 2025. Comparative Analysis of Deep Learning Models for Real-World ISP Network Traffic Forecasting. arXiv preprint arXiv:2503.17410."](https://arxiv.org/abs/2503.17410).

| Benchmark hash | Dataset | Aggregation | Source |
|:-----------------|:-----------------:|:-----------------:|:-----------------:|
[2439d12c2292][2439d12c2292] | CESNET-TimeSeries24 | 1 HOUR| INSTITUTIONS |
[63882fe052f8][63882fe052f8] | CESNET-TimeSeries24 | 1 HOUR| INSTITUTIONS |
[5a79f6cd3506][5a79f6cd3506] | CESNET-TimeSeries24 | 1 HOUR| INSTITUTIONS |
[5a7471b2c70b][5a7471b2c70b] | CESNET-TimeSeries24 | 1 HOUR| INSTITUTIONS |
[0f4fbc0419ce][0f4fbc0419ce] | CESNET-TimeSeries24 | 1 HOUR| INSTITUTIONS |
[d166d3b19a87][d166d3b19a87] | CESNET-TimeSeries24 | 1 HOUR| INSTITUTION_SUBNETS |
[2112383abab7][2112383abab7] | CESNET-TimeSeries24 | 1 HOUR| INSTITUTION_SUBNETS |
[5d9e6e63cbf0][5d9e6e63cbf0] | CESNET-TimeSeries24 | 1 HOUR| INSTITUTION_SUBNETS |
[d82139cd671f][d82139cd671f] | CESNET-TimeSeries24 | 1 HOUR| INSTITUTION_SUBNETS |
[d03ba8db5892][d03ba8db5892] | CESNET-TimeSeries24 | 1 HOUR| INSTITUTION_SUBNETS |
[2e92831cb502][2e92831cb502] | CESNET-TimeSeries24 | 1 HOUR| IP_ADDRESSES_SAMPLE |
[702e58166879][702e58166879] | CESNET-TimeSeries24 | 1 HOUR| IP_ADDRESSES_SAMPLE |
[394603854070][394603854070] | CESNET-TimeSeries24 | 1 HOUR| IP_ADDRESSES_SAMPLE |
[420d4303f949][420d4303f949] | CESNET-TimeSeries24 | 1 HOUR| IP_ADDRESSES_SAMPLE |
[e2c2148a178c][e2c2148a178c] | CESNET-TimeSeries24 | 1 HOUR| IP_ADDRESSES_SAMPLE |

Example of usage of this related works configs:

```python
from cesnet_tszoo.benchmarks import load_benchmark
from cesnet_tszoo.utils.enums import FillerType, ScalerType

benchmark = load_benchmark("2439d12c2292", "../")
dataset = benchmark.get_initialized_dataset()

# Get related results
related_results = benchmark.get_related_results()
print(related_results)

# Process with your own defined model
results = []
for ts_id in tqdm.tqdm(dataset.get_data_about_set(about='train')['ts_ids']):
    model = SimpleLSTM().to(device)
    model.fit(
        dataset.get_train_dataloader(ts_id), 
        dataset.get_val_dataloader(ts_id), 
        n_epochs=5, 
        device=device,
    )
    y_pred, y_true = model.predict(
        dataset.get_test_dataloader(ts_id), 
        device=device,
    )
    
    rmse = mean_squared_error(y_true, y_pred)
    results.append(rmse)

_mean = round(np.mean(results), 3)
_std = round(np.std(results), 3)
print(f"Mean RMSE: {_mean}")
print(f"Std RMSE: {_std}") 

# Compare with related works results
better_works = related_results[related_results['Avg. RMSE'] < _mean]
worse_works = related_results[related_results['Avg. RMSE'] <= _mean]
print(better_works)
print(worse_works)
```
