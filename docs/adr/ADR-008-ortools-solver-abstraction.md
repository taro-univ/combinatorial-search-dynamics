# ADR-008: Taskと参照Solverを分離し、CP-SATを採用する

## Status

Accepted — 2026-09-12。ADR-007はOptunaのために予約する。

## Context

Phase 2のdummy問題は最小化であり、Phase 3の0-1ナップサックは最大化である。探索軌道を作るmock generatorと真値の参照solverは別の責務を持つ。参照解が最適と証明されない場合、探索の成功や真の最適性gapを断定できない。

## Decision

Taskは状態、合法action、feasibility、目的関数の向きと比較を定義する。Solverは独立したProtocolとimmutable結果を持ち、OR-Tools固有型をpipelineへ漏らさない。最初の実装はOR-Tools CP-SATの整数0-1モデルで、CPUの`num_search_workers=1`を既定にしてseedとともに再現性を優先する。solverの解はTaskでfeasibilityと目的値を再計算する。

`OPTIMAL`の場合だけ`optimal_value`を保存する。`FEASIBLE`は`best_feasible_value`と`best_bound`を保持し、`optimal_value`はnullとする。最大化で実行可能値を`z`、上界を`b`とすると、相対gapは`max(0, b-z) / max(abs(z), epsilon)`、既定`epsilon=1e-9`である。解がないときはgapもnull、最適性証明時は0とする。timeoutは`FEASIBLE`または`UNKNOWN`かつ実測経過が設定時間の95%以上の場合に分類し、`timed_out`とstatusを分けて行として残す。solver errorも削除しない。最適性未証明の参照からcheckpointの真のgapやsuccessを作らない。

## Alternatives

mock generatorを参照solverとして兼用する案は、探索と真値を混同するため採用しない。全探索は小規模でも拡張性が低い。複数workerは速い場合があるが、pilotでは再現性を優先する。

## Consequences

将来のsolverは同じ結果interfaceとregistryへ追加できる。timeout時は参照が利用できないinstanceが残り得る。全instanceで最適性が証明されなければgap状態モデルのfitは明確に失敗する。Phase 1の4つの必須schemaとversion `1`は変えず、`reference_solutions`を任意拡張表として追加する。
