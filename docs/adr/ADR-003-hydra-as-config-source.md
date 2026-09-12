# ADR-003 Hydraを設定の正本とする

## Status
Accepted

## Context
研究条件をコードに埋め込むと、実行条件の比較と再現が難しくなる。秘密情報は研究条件と分離する必要がある。

## Decision
Hydraの`configs/`を研究条件の正本とし、分類別のdefault compositionを使う。実行時には解決済み設定を取得できるようにする。秘密情報はYAMLへ保存しない。

## Alternatives
Python定数、単一YAML、環境変数だけで全条件を管理する方式を検討した。

## Consequences
設定差分とoverrideが明示される。各実装は設定を引数として受け取る必要がある。秘密情報は将来のadapterが環境から別途取得する。
