# ADR-006 Git、DVC、MLflowの責務

## Status

Accepted

## Context

Phase 2のpilotは、条件、生成データ、再計算グラフ、学習artifact、run記録を扱う。全てを一箇所へ複製すると、正本と更新契機が曖昧になる。CIと開発環境では外部サービスを使用しない。

## Decision

Gitはコード、Hydra設定、文書、`dvc.yaml`、`dvc.lock`とDVC metadataを管理する。Hydraが科学条件とpathの正本であり、`params.yaml`に同じ値を置かない。DVCは生成データのcacheとstage依存関係を管理し、Gitへ生成ファイルを入れない。`dvc.yaml`内のpathはHydra storage設定の既定値への投影とする。MLflowはlocal file storeでrun、parameter、metric、tag、artifactの参照を記録し、再集計可能なParquet、JSON、CSV、Markdownもそれぞれ保存する。serverとremoteは構成しない。MLflow 3.16以降のfile-store明示opt-inはtracking adapter内で有効にする。credentialをGitへ保存しない。

## Alternatives

MLflowやDVCを科学設定の正本にする案、永続DBやremoteを初期導入する案、全生成物をGit管理する案を検討した。

## Consequences

remoteなしでCPU pilotを再実行でき、2回目の`dvc repro`は未変更stageを省く。Hydra storage既定pathを変更する際は`dvc.yaml`のpathも同時に更新する。将来のremote導入時にはcredentialを別管理し、この責務境界を保つ。DVC revisionを取得できない段階では`unavailable`と記録し、架空のrevisionを発行しない。
