PYTHON_INTERPRERER = python 
CONDA_ENV ?= news_trade
export PYTHONPATH=$(PWD):$PYTHONPATH

create_conda_env:
	@echo "Creating Conda Environemnt"
	conda env create file=env.yml

run_app:
	streamlit run src/app/Newsfeed.py 