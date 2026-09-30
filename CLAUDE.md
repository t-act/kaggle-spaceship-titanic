# kaggle-spaceship-titanic

Kaggle Spaceship Titanic コンペの実験リポジトリ。ローカルで学習し、Kaggle CLI で提出する。

## 役割分担

- 人間: 仮説の立案、実験方針の決定、採否の判断
- Claude Code: 実装、学習の実行、結果の記録、ダッシュボードの更新

仮説や方針を独断で変えない。改善案は NOTES.md の「提案」に書き、人間の判断を待つ。

## 構成

- `src/`: 全実験で共通のコード。データ読み込み（`data.py`）、CV 分割（`cv.py`）、評価関数（`metric.py`）、results.json のスキーマ（`results.py`）、実行記録（`experiment.py`）
- `experiments/<exp_id>/`: 1実験1ディレクトリ。`config.yaml` と `train.py` を置き、実行すると `results.json` が出る。`oof.npy` と `test_proba.npy` はアンサンブルの入力になる（git 管理外）
- `scripts/`: `validate_results.py`（検証）、`build_dashboard.py`（HTML 生成）、`new_experiment.py`（複製）、`sync_lb.py`（LB の記入）
- `dashboard/index.html`: 生成物。ブラウザで直接開く
- `NOTES.md`: 仮説、結論、次の一手、提案。セッションを跨いで文脈を保つ
- `old_notebooks/`: 移行前の notebook。編集しない

## 実験のルール

1. 既存の実験ディレクトリを上書きしない。変更するときは `make new FROM=<親> ORIGIN=<起案者>` で新しい番号に複製する。実行済みの実験は `Experiment` が再実行を拒否する
2. 人間の起案は `exp001` 形式、Claude Code の起案は `expA001` 形式。人間が指示した内容をそのまま実装する場合は human、Claude Code が自分で考えた施策は claude
3. CV 分割は `src/cv.py` の `get_repeats()` だけを使い、実験ごとに変えない。分割は PassengerId のグループ単位の 5 fold を、分割 seed 3つで繰り返す。OOF は分割ごとに持ち、`repeated_fold_scores()` で採点する。方式名は `results.json` の `cv.scheme` に残る。方式が違う実験の CV は比較しない
4. 実行は `make train EXP=<exp_id>`。学習、検証、ダッシュボード更新までを1回で行う
5. 実行後、`results.json` の `conclusion` に所見を書き、NOTES.md に仮説、結果の解釈、次の一手を追記する。採否は人間が追記する
6. 実行前にコミットする。未コミットの変更があると `git_hash` に `-dirty` が付く
7. `lb` は `make sync-lb` で提出履歴から書き写す。手では書き換えない
8. 検証 fold は予測にだけ使う。early stopping や最良時点のモデル選択は、学習側をさらに分けた内側の検証で行う

## コマンド

- `make new FROM=exp001 ORIGIN=claude`: 実験を複製する
- `make train EXP=exp001`: 学習して検証し、ダッシュボードを更新する
- `make validate` / `make dashboard`: 検証、ダッシュボード生成を単独で行う
- `make submit EXP=exp001`: `experiments/exp001/submission.csv` を提出する（メッセージは実験 ID）
- `make sync-lb`: 提出履歴のスコアを各実験の `lb` に書き写し、ダッシュボードを更新する
- `make submissions`: 提出履歴とスコアを表示する

提出は1日の回数制限があり外部に記録が残るため、人間の指示があるときだけ行う。
