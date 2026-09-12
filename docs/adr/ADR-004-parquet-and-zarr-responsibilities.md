# ADR-004 ParquetとZarrの責務分離

## Status
Accepted

## Context

試行メタデータと多次元観測では、必要なschema検査、アクセス単位、欠損表現が異なる。Zarr format versionも、現在の依存環境で読み書きを検証して固定する必要がある。

## Decision

表形式データはPyArrow schemaを正本としてParquetへZstandard圧縮で保存する。多次元観測はZarrへ保存し、valuesとvalid_maskを分離する。両者はcheckpoint ID、trial ID、生成token位置で対応付ける。

Zarr 3.1.6でv3 store、配列属性、可変長UTF-8 indexのround-tripをPython 3.11上で確認できたため、Zarr format 3を採用する。root attributesにschema version、format、完了状態、model・tokenizer revision、観測コードversionを保存する。

## Alternatives

すべてをParquetへ保存する方式、テンソルをParquetへ埋め込む方式、Zarr v2を検討した。Zarr v2は成熟しているが、現在固定したZarr 3系でv3の必要APIを安定して検証でき、将来形式を最初から採用できるため選ばなかった。

## Consequences

表の検索性と配列のchunkアクセスを分離できる。一方、二つの保存形式は明示的IDで整合性を検証する必要がある。Zarr v3または文字列dtypeの互換性に問題が生じた場合は、新しいADRとschema versionで変更し、既存storeを黙って読み替えない。
