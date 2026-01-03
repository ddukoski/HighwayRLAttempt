
.PHONY: install train-highway train-parking eval-highway eval-parking plot archive clean

install:
	python3 -m pip install -r requirements.txt

train-highway:
	python3 training.py

train-parking:
	python3 parking.py

eval-highway:
	python3 training.py --eval

eval-parking:
	python3 parking.py --eval

plot:
	python3 -m scripts.tensorboard_plot --logdir ./logs --outdir ./logs/plots --poll 0

archive:
	python3 -m scripts.keep_best_run --logdir ./logs

clean:
	rm -rf __pycache__ .pytest_cache *.pyc logs/*.png logs/plots logs/*.zip

