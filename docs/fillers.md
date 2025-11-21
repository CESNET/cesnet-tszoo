# Fillers {#cesnet_tszoo.general.fillers}

The `cesnet_tszoo` package supports various ways of dealing with missing data in dataset.
Possible config parameters in [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig], [`DisjointTimeBasedConfig`][cesnet_tszoo.configs.disjoint_time_based_config.DisjointTimeBasedConfig] and [`SeriesBasedConfig`][cesnet_tszoo.configs.series_based_config.SeriesBasedConfig]:

- `fill_missing_with`: Can pass enum [`FillerType`][cesnet_tszoo.utils.enums.FillerType] for built-in filler or pass a type of custom filler that must derive from [`Filler`][cesnet_tszoo.utils.filler.filler.Filler] base class.
- `default_values`: Default values for missing data, applied before fillers. Can set one value for all features or specify for each feature.

!!! info "Note for [`TimeBasedConfig`][cesnet_tszoo.configs.time_based_config.TimeBasedConfig]"
    Fillers can carry over values from the train set to the validation and test sets. For example, [`ForwardFiller`][cesnet_tszoo.utils.filler.filler.ForwardFiller] can carry over values from previous sets. 

### Built-in fillers
The `cesnet_tszoo` package comes with multiple built-in fillers. To check built-in fillers refer to [`fillers`][cesnet_tszoo.utils.filler.filler].

### Custom fillers
It is possible to create and use own fillers. But custom filler must derive from [`Filler`][cesnet_tszoo.utils.filler.filler.Filler] base class.