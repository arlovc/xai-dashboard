#!/usr/bin/env bash
source ~/.bashrc
export PATH=~/anaconda3/bin:$PATH

#reads input from command line into a specific variable
read -p "Create new conda env (y/n)?" NEWENV
# read -p "Use custom argument.txt names (y/n)?" CUSTOM
conda init bash

mapfile -t <arguments.txt


if [ "$NEWENV" == "y" ]; then
    # user chooses to create conda env
    # prompt user for conda env name
    echo "Creating new conda environment named xai_dashboard8"
    # create conda environment with .yaml file
    conda env create -f environment.yml

else
    echo "No new environment created, running dashboard"
fi

conda activate xai_dashboard8
streamlit run src/streamlit_app.py
