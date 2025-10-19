run:
	python main.py

train:
	python main.py --train

clean:
	rm -rf __pycache__ *.pyc logs *.zip
