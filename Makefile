
.PHONY: install train-highway train-parking eval-highway eval-parking plot archive clean

install:
	python3 -m pip install -r requirements.txt

train:
	python3 training.py

eval:
	python3 training.py --eval

plot:
	python3 -m scripts.tensorboard_plot --logdir ./logs --outdir ./logs/plots --poll 0

archive:
	python3 -m scripts.keep_best_run --logdir ./logs

clean:
	rm -rf __pycache__ .pytest_cache *.pyc logs/*.png logs/plots logs/*.zip

