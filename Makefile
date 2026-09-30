COMPETITION := spaceship-titanic
NB ?= 03_optimaze_model
MSG ?= $(NB)

.PHONY: setup data lab run submit submissions

setup:
	uv sync

data:
	uv run kaggle competitions download -c $(COMPETITION) -p data
	cd data && unzip -o $(COMPETITION).zip && rm $(COMPETITION).zip

lab:
	uv run jupyter lab

# notebook をヘッドレスで実行し、出力をその notebook に書き戻す
run:
	cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace $(NB).ipynb

submit:
	uv run kaggle competitions submit -c $(COMPETITION) -f submissions/submission.csv -m "$(MSG)"

submissions:
	uv run kaggle competitions submissions -c $(COMPETITION)
