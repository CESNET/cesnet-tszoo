# CESNET-TimeSeries26 { #cesnet_timeseries26_page }

!!! info "Release status"
    CESNET-TimeSeries26 is the successor of [CESNET-TimeSeries24][cesnet_timeseries24_page]. The dataset is published
    on Zenodo, https://zenodo.org/records/21945629 (DOI [10.5281/zenodo.21945629](https://doi.org/10.5281/zenodo.21945629)).
    All 30 HDF5 files of the library (10 sources × 3 aggregations) are downloaded from the sources listed under
    *Download sources* below. The dataset paper is in preparation; until it is out, please cite the dataset record and
    the CESNET-TimeSeries24 paper.

#### Data capture

The dataset was collected by long-term IP-flow monitoring of the ISP network CESNET3 (Czech Education and Science
Network), with the same statistical metrics as CESNET-TimeSeries24. The main collection covers
**9 October 2023 – 30 June 2026** (996 days, ~2.7 years; 140 636 ten-minute, 23 902 hourly and 996 daily slots);
the `subnet_longitudinal` source follows the IP addresses of a single institution subnet from
**31 December 2021 to 9 July 2026** (4.5 years) on its own, longer time grid. Extending the 40-week
CESNET-TimeSeries24 collection to a multi-year span makes annual seasonality, year-over-year drift and model-longevity
studies possible.

#### Data description

Datapoints are created by aggregating IP flow records into time windows of 10 minutes, 1 hour and 1 day. Each
datapoint is multivariate and carries the same metric families as CESNET-TimeSeries24:

- **Simple volumetric metrics:** the number of IP flows, the number of packets, and the transmitted data size (i.e. number of bytes)
- **Unique volumetric metrics:** the numbers of unique destination IP addresses, ASNs and transport-layer ports.
  Wherever several 10-minute windows are combined (coarser aggregations, institution and subnet aggregates) they are
  given as sum / average / standard deviation over the constituent 10-minute windows
- **Ratio metrics:** the ratio of UDP/TCP packets, the ratio of UDP/TCP transmitted data size, the direction ratio of packets, and the direction ratio of transmitted data size
- **Average metrics:** the average flow duration, and the average Time To Live (TTL)

Only the windows in which the entity was actually observed are stored, so series differ in length and most IP-level
series are sparse (the median IP series is present in roughly 5 % of the 10-minute slots, 14 % of the hourly slots
and 58 % of the daily slots). The library fills the missing windows on request (`fill_missing_with`).

#### Sources

| Source type (`source_type`)        | Time series | Identifier              | Period                  | Description |
|------------------------------------|------------:|-------------------------|-------------------------|-------------|
| `institutions`                     | 285         | `id_institution`        | 2023-10-09 – 2026-06-30 | Traffic aggregated per institution. |
| `institution_subnets`              | 566         | `id_institution_subnet` | 2023-10-09 – 2026-06-30 | Traffic aggregated per institution subnet. |
| `ip_addresses_full`                | 278 466     | `id_ip`                 | 2023-10-09 – 2026-06-30 | Every monitored IP address. Large files: 87.6 GB (10 minutes; 5.46 G rows), 46.0 GB (1 hour), 5.7 GB (1 day). |
| `ip_addresses_sample`              | 1 000       | `id_ip`                 | 2023-10-09 – 2026-06-30 | Uniform random sample of `ip_addresses_full` (as in CESNET-TimeSeries24). |
| `ip_addresses_sample_gaps_20_00`   | 1 000       | `id_ip`                 | 2023-10-09 – 2026-06-30 | Gap-stratified sample, gap ratio [0.00, 0.20] — near-complete to fully dense series. |
| `ip_addresses_sample_gaps_40_20`   | 1 000       | `id_ip`                 | 2023-10-09 – 2026-06-30 | Gap ratio (0.20, 0.40] — mostly complete series. |
| `ip_addresses_sample_gaps_60_40`   | 1 000       | `id_ip`                 | 2023-10-09 – 2026-06-30 | Gap ratio (0.40, 0.60] — moderately gapped series. |
| `ip_addresses_sample_gaps_80_60`   | 1 000       | `id_ip`                 | 2023-10-09 – 2026-06-30 | Gap ratio (0.60, 0.80] — sparse series. |
| `ip_addresses_sample_gaps_100_80`  | 1 000       | `id_ip`                 | 2023-10-09 – 2026-06-30 | Gap ratio (0.80, 1.00] — very sparse series. |
| `subnet_longitudinal`              | 135         | `id_ip`                 | 2021-12-31 – 2026-07-09 | IP addresses of one institution subnet monitored for 4.5 years (own time grid, reduced metric set, see below). |

Every source is available with the aggregations `10_minutes`, `1_hour` and `1_day`.

**Gap-stratified samples.** For every IP series the *gap ratio* is computed on the 10-minute aggregation as
`gap_ratio = 1 − n_present / 143 280`, where `n_present` is the number of 10-minute windows in which the series has
an observation and 143 280 = 995 × 144 is the fixed nominal number of 10-minute windows of the collection period
(the denominator is the same constant for every series, so a series active only for part of the period counts as
gapped). The five `ip_addresses_sample_gaps_*` sources are disjoint samples of 1 000 series from the five equidistant
bands of the gap ratio (upper bound inclusive, the bottom band closed at 0), so the effect of missing data on a model
can be studied per band; `ip_addresses_sample` is an independent uniform random sample and may overlap with them.
The samples are copies of the corresponding `ip_addresses_full` series, so results carry over unchanged.

#### List of time series metrics

Column availability: **A** = 10-minute files of the IP-address sources (`ip_addresses_full`, `ip_addresses_sample`,
`ip_addresses_sample_gaps_*`); **B** = 1-hour and 1-day files of the IP-address sources and *all* aggregations of
`institutions` and `institution_subnets`; **C** = `subnet_longitudinal` (all aggregations).

|  Time Series Metric    | A | B | C | Description                                                                |
|------------------------|:-:|:-:|:-:|----------------------------------------------------------------------------|
| id_time                | ✓ | ✓ | ✓ | Unique identifier for each aggregation interval within the time series.      |
| n_flows                | ✓ | ✓ | ✓ | Total number of flows observed in the aggregation interval.                  |
| n_packets              | ✓ | ✓ | ✓ | Total number of packets transmitted during the aggregation interval.         |
| n_bytes                | ✓ | ✓ | ✓ | Total number of bytes transmitted during the aggregation interval.           |
| n_dest_ip              | ✓ |   | ✓ | Number of unique destination IP addresses.                                   |
| n_dest_asn             | ✓ |   | ✓ | Number of unique destination ASNs.                                           |
| n_dest_ports           | ✓ |   |   | Number of unique destination transport layer ports.                          |
| n_dest_port            |   |   | ✓ | As `n_dest_ports` (note the singular name in `subnet_longitudinal`).         |
| sum_n_dest_ip          |   | ✓ |   | Sum of the 10-minute numbers of unique destination IP addresses.             |
| average_n_dest_ip      |   | ✓ |   | The average number of unique destination IP addresses.                       |
| std_n_dest_ip          |   | ✓ |   | Standard deviation of numbers of unique destination IP addresses.            |
| sum_n_dest_asn         |   | ✓ |   | Sum of numbers of unique destination ASNs.                                   |
| average_n_dest_asn     |   | ✓ |   | The average number of unique destination ASNs.                               |
| std_n_dest_asn         |   | ✓ |   | Standard deviation of numbers of unique destination ASNs.                    |
| sum_n_dest_ports       |   | ✓ |   | Sum of numbers of unique destination transport layer ports.                  |
| average_n_dest_ports   |   | ✓ |   | The average number of unique destination transport layer ports.              |
| std_n_dest_ports       |   | ✓ |   | Standard deviation of numbers of unique destination transport layer ports.   |
| tcp_udp_ratio_packets  | ✓ | ✓ |   | Ratio of packets sent using TCP versus UDP, in the interval <0, 1> (1 = all TCP, 0 = all UDP). |
| tcp_udp_ratio_bytes    | ✓ | ✓ |   | Ratio of bytes sent using TCP versus UDP, same rule as tcp_udp_ratio_packets. |
| dir_ratio_packets      | ✓ | ✓ | ✓ | Ratio of packet directions, in the interval <0, 1> (1 = all outgoing, 0 = all incoming). |
| dir_ratio_bytes        | ✓ | ✓ | ✓ | Ratio of byte directions, same rule as dir_ratio_packets.                    |
| avg_duration           | ✓ | ✓ | ✓ | Average duration of IP flows during the aggregation interval, in seconds.    |
| avg_ttl                | ✓ | ✓ |   | Average Time To Live (TTL) of IP flows during the aggregation interval.      |

Notes:

- The re-aggregated unique metrics are named `average_*` (CESNET-TimeSeries24 uses `avg_*`). The standard deviation
  is the sample standard deviation (ddof = 1) and is 0 where a window contains a single 10-minute datapoint. Numeric
  values are rounded to two decimals.
- The 1-hour and 1-day files of the main sources are deterministic re-aggregations of the 10-minute files (counters
  summed, cardinalities as sum/average/std, ratio and average columns as unweighted means).
- `subnet_longitudinal` is only partly consistent with the rest: its volumetric counters cover **TCP traffic only**,
  it has neither the `tcp_udp_ratio_*` columns nor `avg_ttl`, its `id_time` values live on its own time grid and are
  **not comparable** with the other sources, and its 1-hour and 1-day files are not exact re-aggregations of its
  10-minute file. Use it for long-horizon analysis of that one subnet, not for cross-source comparison.

!!! warning "Library version"
    The 10-minute `ip_addresses_full` table has more than 2³¹ rows. Library versions before this fork's fix of
    2026-09-10 (`pytables_data/base_datasets/*.py`, int32 row-offset arithmetic) fail with `OverflowError` for
    about three quarters of its series; use this fork (or a release that includes the fix) for that file.

#### Known properties of the data

These are properties of operational data, not defects:

- **The time grid is not contiguous.** Network-wide outages are absent from the grid itself (140 636 instead of
  143 412 nominal ten-minute slots: 2 718 slots of the seven documented outages and 58 further slots of short
  undocumented gaps), so always resolve `id_time` through the times table (`ds.metadata.time_indices`) rather
  than assuming a fixed step. The hourly aggregation keeps every hour slot; an hour overlapping an outage holds the
  sum of the ten-minute windows that exist (0 when none does), not a measured hour.
- **Entity lifetimes are right-censored.** Addresses were selected by activity within the collection window, so almost
  every series is present at the end of the collection; end dates carry no information about when an address ceased
  to be used.
- **A late-November-2023 onboarding step.** About 34 700 IP series first appear on 2023-11-30 and network-wide
  active-address counts step up around 2023-11-25 — a monitoring expansion, not a traffic change.
- **Daylight saving.** At spring-forward transitions two measurement windows fall on one grid slot and are merged;
  at autumn transitions the local clock repeats an hour. Work in UTC or use the `europe_time_changes` table.

#### Additional data

The following small tables are available through `get_additional_data`:

| Table                              | Content                                                                             |
|------------------------------------|-------------------------------------------------------------------------------------|
| weekends_and_holidays              | Dates of weekends and Czech public holidays covering the collection period (515 rows). |
| school_terms_and_exams_periods     | Approximate Czech academic terms and examination periods (19 rows; summer recess is the gap between `SummerExams` and the next `WinterTerm`). |
| europe_time_changes                | Daylight-saving transitions (9 rows) with direction, local clock change, UTC offsets before/after and the signed offset change. |
| outages                            | Documented monitoring outages (13 rows): 7 infrastructure-wide grid gaps in the main collection (the largest 2026-04-29 – 2026-05-15, ~15.9 days) and 6 subnet-local zero-traffic periods in `subnet_longitudinal`; with scope, kind, last data before / data resumed, duration and number of 10-minute windows. |
| ids_relationship                   | Mapping of each of the 278 466 `id_ip` values to its `id_institution` and `id_institution_subnet`. |

Entity ids are pseudonymous integers; no IP addresses, institution names or other identifying information are published.

#### Download sources

`CESNET_TimeSeries26` tries its sources in a fixed order, defined in
`cesnet_tszoo/datasets/databases/cesnet_timeseries26/constants.py`, and verifies the SHA-256 digest of every file
(`CHECKSUMS`) after download. A source that answers with an error, an HTML page or a file with a different digest is
skipped and the next one is tried; partial downloads are resumed. Pass `download_source="datazoo"`,
`"google_drive"` or `"zenodo"` to `get_dataset` to use one source only.

1. **CESNET DataZoo**, the same bucket as CESNET-TimeSeries24 (`BUCKET_URL`). The files are not uploaded there yet;
   until they are, this request fails and the library moves on without user action.
2. **Google Drive**, primary folder `1v7b-IDkIHIFpNZfSSLurpOaEsxCGy4x_` (`DOWNLOAD_URLS`). Drive may temporarily
   refuse very large files ("too many users have viewed or downloaded this file recently"); the next source is then
   tried.
3. **Google Drive**, backup folder (`BACKUP_DRIVE_IDS`, filled with `scripts/ts26_backup_drive_ids.py` once the files
   are uploaded).
4. **Zenodo** (persistent, checksummed; `ZENODO_SOURCES`): the 30 files are spread over four records,
   [23278901](https://zenodo.org/records/23278901), [23280650](https://zenodo.org/records/23280650),
   [23280754](https://zenodo.org/records/23280754) and [23283552](https://zenodo.org/records/23283552)
   (main dataset record: [21945629](https://zenodo.org/records/21945629)). Files larger than 500 MB are split into
   500 MB parts (`<file>.h5.001`, `.002`, …); the library downloads the parts of a file in parallel (4 at once, set
   `CESNET_TSZOO_PARALLEL_DOWNLOADS` to change it) and re-assembles them transparently (`parts_source`). Without the
   library, concatenate the parts with `cat` in order and compare the SHA-256 of the result with `CHECKSUMS`; every
   record also contains `SHA256SUMS.txt` with the digests of its parts.

### Links

Contact dataset author [Josef Koumar](https://koumajos.github.io/). Previous version of the dataset: [CESNET-TimeSeries24][cesnet_timeseries24_page].
