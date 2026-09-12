# ADR-002 src layout

## Status
Accepted

## Context
作業ディレクトリから偶然importできる状態を避け、インストールされたパッケージをテストしたい。

## Decision
Pythonパッケージを`src/llm_search_dynamics`以下に配置し、Hatchlingでビルドする。

## Alternatives
リポジトリ直下にパッケージを置くflat layoutを検討した。

## Consequences
テストとCLIはインストール済みパッケージを通る。新しい再利用コードは`src/`へ置く必要がある。
