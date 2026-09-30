COMPETITION := spaceship-titanic

.PHONY: setup data lab new train validate dashboard submit submissions

setup:
	uv sync

data:
	uv run kaggle competitions download -c $(COMPETITION) -p data
	cd data && unzip -o $(COMPETITION).zip && rm $(COMPETITION).zip

lab:
	uv run jupyter lab

# 既存の実験を新しい番号で複製する。例: make new FROM=exp001 ORIGIN=claude
new:
	@test -n "$(FROM)" -a -n "$(ORIGIN)" || { echo "例: make new FROM=exp001 ORIGIN=human|claude"; exit 1; }
	uv run python scripts/new_experiment.py --from $(FROM) --origin $(ORIGIN)

# 学習 → results.json の検証 → ダッシュボード更新までを1回で行う。例: make train EXP=exp001
train:
	@test -n "$(EXP)" || { echo "例: make train EXP=exp001"; exit 1; }
	uv run python experiments/$(EXP)/train.py
	$(MAKE) validate dashboard

validate:
	uv run python scripts/validate_results.py

dashboard:
	uv run python scripts/build_dashboard.py

# 提出メッセージに実験 ID を入れ、LB と実験を後から突き合わせられるようにする。例: make submit EXP=exp001
submit:
	@test -n "$(EXP)" || { echo "例: make submit EXP=exp001"; exit 1; }
	uv run kaggle competitions submit -c $(COMPETITION) -f experiments/$(EXP)/submission.csv -m "$(EXP)"

submissions:
	uv run kaggle competitions submissions -c $(COMPETITION)
