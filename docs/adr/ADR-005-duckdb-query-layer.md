# ADR-005 DuckDBを問合せ層として使用する

## Status
Accepted

## Context

複数のParquetを横断分析したいが、同じデータを別の永続databaseへ複製すると正本が曖昧になる。checkpoint分析ではinstanceとtrialの条件を明示的IDで結ぶ必要がある。

## Decision

DuckDBはin-memoryの問合せ層として使用し、4つのParquetを直接viewとして参照する。`analysis_checkpoints`は`instance_id`と`trial_id`でinstances、trials、checkpointsをjoinする。metricsは独立viewとし、代表viewへ含めない。

## Alternatives

永続DuckDB databaseを正本にする方式、データをDuckDB tableへcopyする方式、各分析で個別にPyArrow joinする方式を検討した。

## Consequences

Parquetを唯一の表データ正本として保ちながらSQL分析できる。問合せ前に必要ファイルとschemaを検証する負担がある。将来永続viewやcatalogが必要になった場合も、データ本体の所有権は別途ADRで決定する。
