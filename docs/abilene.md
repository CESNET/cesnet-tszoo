# Abilene { #abilene_page }

#### Data capture

The data was captured in the backbone infrastructure of the Abilene Network operated by Internet2. The capturing was done over several months in 2004, during which traffic measurements were recorded at regular 5-minute intervals from core network links.

#### Data description

The dataset consists of aggregated traffic matrices representing communication between backbone nodes of the Abilene network. The created datapoints represent a summarized traffic volumes between origin–destination node pairs within a defined time window of 5 minutes.

The dataset is divided into three subsets: **Matrix**, **Node2Node**, and **Node**. Each subset is additionaly aggregated into 10-minute, 1-hour, and 1-day time window intervals.

### Matrix subset
In this subset, features are 12x12 TMs. Contains 1 time series.

#### 5-minutes interval time series metrics

| Time Series Metric              | Description                                                                                             |
|---------------------------------|---------------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| matrix_realOD                   | The measured traffic volumes between all origin–destination (OD) node pairs in the Abilene network.    |
| matrix_simpleGravityOD          | The estimated OD traffic matrix using a simple gravity model applied to Abilene backbone measurements. |
| matrix_generalGravityOD         | The estimated OD traffic matrix using a generalized gravity model applied to Abilene backbone measurements. |
| matrix_simpleTomogravityOD      | The estimated OD traffic matrix using a simple tomogravity model applied to Abilene backbone measurements. |
| matrix_generalTomogravityOD     | The estimated OD traffic matrix using a generalized tomogravity model applied to Abilene backbone measurements. |

#### 10-minutes, 1-hour and 1-day interval time series metrics

| Time Series Metric                 | Description                                                                                                      |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| matrix_avg_realOD                  | The average of measured OD traffic volumes in the Abilene network over the aggregation interval.                |
| matrix_std_realOD                  | The standard deviation of measured OD traffic volumes in the Abilene network over the aggregation interval.     |
| matrix_sum_realOD                  | The sum of measured OD traffic volumes in the Abilene network over the aggregation interval.                     |
| matrix_avg_simpleGravityOD         | The average of OD traffic volumes estimated by the simple gravity model over the aggregation interval.           |
| matrix_std_simpleGravityOD         | The standard deviation of OD traffic volumes estimated by the simple gravity model over the aggregation interval.|
| matrix_sum_simpleGravityOD         | The sum of OD traffic volumes estimated by the simple gravity model over the aggregation interval.               |
| matrix_avg_generalGravityOD        | The average of OD traffic volumes estimated by the generalized gravity model over the aggregation interval.      |
| matrix_std_generalGravityOD        | The standard deviation of OD traffic volumes estimated by the generalized gravity model over the aggregation interval.|
| matrix_sum_generalGravityOD        | The sum of OD traffic volumes estimated by the generalized gravity model over the aggregation interval.          |
| matrix_avg_simpleTomogravityOD     | The average of OD traffic volumes estimated by the simple tomogravity model over the aggregation interval.       |
| matrix_std_simpleTomogravityOD     | The standard deviation of OD traffic volumes estimated by the simple tomogravity model over the aggregation interval.|
| matrix_sum_simpleTomogravityOD     | The sum of OD traffic volumes estimated by the simple tomogravity model over the aggregation interval.           |
| matrix_avg_generalTomogravityOD    | The average of OD traffic volumes estimated by the generalized tomogravity model over the aggregation interval.  |
| matrix_std_generalTomogravityOD    | The standard deviation of OD traffic volumes estimated by the generalized tomogravity model over the aggregation interval.|
| matrix_sum_generalTomogravityOD    | The sum of OD traffic volumes estimated by the generalized tomogravity model over the aggregation interval.      |

### Node2Node subset
In this subset, each time series is a cell from the TMs. Contains 144 time series.

#### 5-minutes interval time series metrics

| Time Series Metric             | Description                                                                                     |
|-------------------------------|-------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| realOD                         | Measured traffic volume between a specific origin–destination (OD) node pair in the Abilene network. |
| simpleGravityOD                | OD traffic volume estimated by the simple gravity model for a specific node pair.              |
| generalGravityOD               | OD traffic volume estimated by the generalized gravity model for a specific node pair.         |
| simpleTomogravityOD            | OD traffic volume estimated by the simple tomogravity model for a specific node pair.          |
| generalTomogravityOD           | OD traffic volume estimated by the generalized tomogravity model for a specific node pair.     |

#### 10-minutes, 1-hour and 1-day interval time series metrics

| Time Series Metric             | Description                                                                                     |
|-------------------------------|-------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| avg_realOD                     | Average traffic volume of a specific OD node pair over the aggregation interval.              |
| std_realOD                     | Standard deviation of traffic volume of a specific OD node pair over the aggregation interval. |
| sum_realOD                     | Sum of traffic volume of a specific OD node pair over the aggregation interval.               |
| avg_simpleGravityOD            | Average traffic volume estimated by the simple gravity model for a specific OD node pair over the aggregation interval. |
| std_simpleGravityOD            | Standard deviation of traffic volume estimated by the simple gravity model for a specific OD node pair. |
| sum_simpleGravityOD            | Sum of traffic volume estimated by the simple gravity model for a specific OD node pair.       |
| avg_generalGravityOD           | Average traffic volume estimated by the generalized gravity model for a specific OD node pair. |
| std_generalGravityOD           | Standard deviation of traffic volume estimated by the generalized gravity model for a specific OD node pair. |
| sum_generalGravityOD           | Sum of traffic volume estimated by the generalized gravity model for a specific OD node pair.   |
| avg_simpleTomogravityOD        | Average traffic volume estimated by the simple tomogravity model for a specific OD node pair.  |
| std_simpleTomogravityOD        | Standard deviation of traffic volume estimated by the simple tomogravity model for a specific OD node pair. |
| sum_simpleTomogravityOD        | Sum of traffic volume estimated by the simple tomogravity model for a specific OD node pair.    |
| avg_generalTomogravityOD       | Average traffic volume estimated by the generalized tomogravity model for a specific OD node pair. |
| std_generalTomogravityOD       | Standard deviation of traffic volume estimated by the generalized tomogravity model for a specific OD node pair. |
| sum_generalTomogravityOD       | Sum of traffic volume estimated by the generalized tomogravity model for a specific OD node pair. |

### Node subset
In this subset, each time series corresponds to one node. Contains 12 time series.

#### Time series metrics

| Time Series Metric              | Description                                                                                      |
|--------------------------------|--------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| avg_from_realOD                 | Average traffic volume originating from a specific node over the aggregation interval.          |
| std_from_realOD                 | Standard deviation of traffic volume originating from a specific node over the aggregation interval. |
| sum_from_realOD                 | Sum of traffic volume originating from a specific node over the aggregation interval.            |
| avg_to_realOD                   | Average traffic volume destined to a specific node over the aggregation interval.               |
| std_to_realOD                   | Standard deviation of traffic volume destined to a specific node over the aggregation interval. |
| sum_to_realOD                   | Sum of traffic volume destined to a specific node over the aggregation interval.                 |
| avg_from_simpleGravityOD        | Average traffic volume estimated by the simple gravity model originating from a node.            |
| std_from_simpleGravityOD        | Standard deviation of traffic volume estimated by the simple gravity model originating from a node. |
| sum_from_simpleGravityOD        | Sum of traffic volume estimated by the simple gravity model originating from a node.             |
| avg_to_simpleGravityOD          | Average traffic volume estimated by the simple gravity model destined to a node.                 |
| std_to_simpleGravityOD          | Standard deviation of traffic volume estimated by the simple gravity model destined to a node.   |
| sum_to_simpleGravityOD          | Sum of traffic volume estimated by the simple gravity model destined to a node.                  |
| avg_from_generalGravityOD       | Average traffic volume estimated by the generalized gravity model originating from a node.       |
| std_from_generalGravityOD       | Standard deviation of traffic volume estimated by the generalized gravity model originating from a node. |
| sum_from_generalGravityOD       | Sum of traffic volume estimated by the generalized gravity model originating from a node.        |
| avg_to_generalGravityOD         | Average traffic volume estimated by the generalized gravity model destined to a node.            |
| std_to_generalGravityOD         | Standard deviation of traffic volume estimated by the generalized gravity model destined to a node. |
| sum_to_generalGravityOD         | Sum of traffic volume estimated by the generalized gravity model destined to a node.             |
| avg_from_simpleTomogravityOD    | Average traffic volume estimated by the simple tomogravity model originating from a node.        |
| std_from_simpleTomogravityOD    | Standard deviation of traffic volume estimated by the simple tomogravity model originating from a node. |
| sum_from_simpleTomogravityOD    | Sum of traffic volume estimated by the simple tomogravity model originating from a node.         |
| avg_to_simpleTomogravityOD      | Average traffic volume estimated by the simple tomogravity model destined to a node.            |
| std_to_simpleTomogravityOD      | Standard deviation of traffic volume estimated by the simple tomogravity model destined to a node. |
| sum_to_simpleTomogravityOD      | Sum of traffic volume estimated by the simple tomogravity model destined to a node.             |
| avg_from_generalTomogravityOD   | Average traffic volume estimated by the generalized tomogravity model originating from a node.   |
| std_from_generalTomogravityOD   | Standard deviation of traffic volume estimated by the generalized tomogravity model originating from a node. |
| sum_from_generalTomogravityOD   | Sum of traffic volume estimated by the generalized tomogravity model originating from a node.    |
| avg_to_generalTomogravityOD     | Average traffic volume estimated by the generalized tomogravity model destined to a node.        |
| std_to_generalTomogravityOD     | Standard deviation of traffic volume estimated by the generalized tomogravity model destined to a node. |
| sum_to_generalTomogravityOD     | Sum of traffic volume estimated by the generalized tomogravity model destined to a node.         |

### Links

More detailed description is available in the [paper](https://arxiv.org/abs/0708.0945). Original dataset can be downloaded at [https://www.cs.utexas.edu/~yzhang/research/AbileneTM](https://www.cs.utexas.edu/~yzhang/research/AbileneTM).
