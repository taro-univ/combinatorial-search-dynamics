# ADR-001 uvとPython 3.11

## Status
Accepted

## Context
開発環境とCIで、Python本体と直接・間接依存を同じ方法で再現する必要がある。

## Decision
Python 3.11を使用し、uvで環境と`uv.lock`を管理する。ローカルとCIは`uv sync --all-extras --locked`で同期する。

## Alternatives
pipとrequirementsファイル、Poetry、Condaを検討した。

## Consequences
依存解決を一元化できる。依存変更時は`uv.lock`も更新する必要がある。Python 3.12以降への移行には本ADRを見直す。
