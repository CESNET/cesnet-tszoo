# Telecom Italia { #telecom_italia_page }

#### Data capture
The data was captured from anonymized mobile network activity collected by Telecom Italia as part of the Telecom Italia Big Data Challenge. Mobile communication events - including SMS sent and received, phone calls incoming and outgoing, and Internet session activity - were aggregated over a grid of square cells covering Milan and surrounding areas. Measurements were recorded at 10‑minute intervals from 1 November 2013 to 1 January 2014, with each event type contributing to the aggregated activity statistics within each cell. The dataset represents spatially localized summaries of communication behavior over time.

#### Data description
The dataset consists of time‑series measurements of telecommunication activity across a grid of spatial cells. Each datapoint corresponds to a 10‑minute time window and includes cell identifiers along with aggregated values proportional to SMS counts (inbound and outbound), call counts (incoming and outgoing), and Internet traffic activity within that cell during the interval. The values reflect aggregated call detail record (CDR) activity, rescaled by a constant factor defined by Telecom Italia for privacy preservation.

Dataset is additionaly aggregated into 1-hour and 1-day time window intervals.

### Time series
Each time series correspond to specific pair of square and country code. Contains 29677 time series. 

Additionaly to prevent lot of empty time series, every pair of (square, country code), which had more than 70% missing values, was merged into one time series with an id 29676.

#### 10-minute interval time series metrics

| Time Series Metric           | Description                                                                                     |
|-------------------------------|-------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| sms_out                  | Number of SMS messages of specific country code sent from the cell during the time interval.                              |
| sms_in                   | Number of SMS messages of specific country code received by the cell during the time interval.                            |
| calls_out                | Number of phone calls of specific country code initiated from the cell during the time interval.                          |
| calls_in                 | Number of phone calls of specific country code received by the cell during the time interval.                             |
| internet_traffic              | Volume of Internet traffic of specific country code (data usage) within the cell during the time interval.                |

#### 1-hour and 1-day interval time series metrics

| Time Series Metric           | Description                                                                                                 |
|-------------------------------|-------------------------------------------------------------------------------------------------------------|
| id_time                         | Unique identifier for each aggregation interval within the time series, used to segment the dataset.   |
| avg_sms_out              | Average number of SMS messages of specific country code sent from the cell over the aggregation interval.                             |
| std_sms_out              | Standard deviation of SMS messages of specific country code sent from the cell over the aggregation interval.                         |
| sum_sms_out              | Sum of SMS messages of specific country code sent from the cell over the aggregation interval.                                        |
| avg_sms_in               | Average number of SMS messages of specific country code received by the cell over the aggregation interval.                            |
| std_sms_in               | Standard deviation of SMS messages of specific country code received by the cell over the aggregation interval.                        |
| sum_sms_in               | Sum of SMS messages of specific country code received by the cell over the aggregation interval.                                      |
| avg_calls_out            | Average number of phone calls of specific country code initiated from the cell over the aggregation interval.                          |
| std_calls_out            | Standard deviation of phone calls of specific country code initiated from the cell over the aggregation interval.                      |
| sum_calls_out            | Sum of phone calls of specific country code initiated from the cell over the aggregation interval.                                     |
| avg_calls_in             | Average number of phone calls of specific country code received by the cell over the aggregation interval.                             |
| std_calls_in             | Standard deviation of phone calls of specific country code received by the cell over the aggregation interval.                         |
| sum_calls_in             | Sum of phone calls of specific country code received by the cell over the aggregation interval.                                       |
| avg_internet_traffic          | Average Internet traffic volume of specific country code within the cell over the aggregation interval.                               |
| std_internet_traffic          | Standard deviation of Internet traffic volume of specific country code within the cell over the aggregation interval.                 |
| sum_internet_traffic          | Sum of Internet traffic volume of specific country code within the cell over the aggregation interval.                                 |

### Links

More detailed description is available in the [paper](https://doi.org/10.1038/sdata.2015.55). Original dataset can be downloaded at [https://dataverse.harvard.edu](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV).
