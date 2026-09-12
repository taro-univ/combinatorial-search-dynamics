# Architecture

Phase 0は、Python 3.11のsrc layout、uvによる依存固定、Hydra設定、CLI検証、テスト、CIからなる骨格である。研究処理はまだ実装しない。

`configs/config.yaml`が設定合成の入口であり、experiment、task、llm、generation、observation、state model、dynamics、evaluation、storageを分類する。各placeholderの`implemented: false`は、その構成が予約済みで実行機能を持たないことを示す。秘密情報は設定に含めない。

`src/llm_search_dynamics/config.py`がHydra設定の合成、解決、およびPhase 0構成検証を担う。`src/llm_search_dynamics/cli.py`はこの機能を`lsd doctor`と`lsd validate-repository`として公開する。

Phase 1以降で、保存、生成、観測、状態モデル、力学、評価をそれぞれ独立したパッケージ境界として追加する。詳細な境界と将来構成は[リポジトリ仕様書](repository-specification.md)を正本とする。
