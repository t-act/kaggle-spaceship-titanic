# kaggle-spaceship-titanic

Kaggle Spaceship Titanic コンペの実験リポジトリ。ローカルで notebook を実行し、Kaggle CLI で提出する。

## 構成

- `notebooks/`: 現行の notebook。番号順に実験が進む
- `old_notebooks/`: 退避した旧 notebook。編集しない
- `data/`: `make data` で取得するコンペデータ（git 管理外）
- `submissions/`: notebook が書き出す `submission.csv`（git 管理外）

notebook は `/kaggle/input` の有無で入出力先を切り替えるため、Kaggle 上でもそのまま動く。

## コマンド

- `make setup`: 依存を入れる（uv）
- `make data`: コンペデータを `data/` に取得
- `make run NB=03_optimaze_model`: notebook をヘッドレス実行
- `make submit MSG="..."`: `submissions/submission.csv` を提出
- `make submissions`: 提出履歴とスコアを表示

提出は1日の回数制限があり外部に記録が残るため、ユーザーの指示があるときだけ行う。
