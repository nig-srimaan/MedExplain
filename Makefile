.PHONY: install train test app clean

install:
	pip install -r requirements.txt

train:
	python3 train_pipeline.py --domain all --folds 5

test:
	python3 -m unittest discover -s tests -v

app:
	streamlit run app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
