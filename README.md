Neuro-EEG — EEGMMIDB + NeuroGPT
==============================

This project prepares the EEGMMIDB motor movement / imagery dataset for use with
NeuroGPT and related deep learning models.


Dataset
-------

EEG Motor Movement/Imagery Dataset (EEGMMIDB):
https://physionet.org/content/eegmmidb/1.0.0/

Download into the `data/` folder:

    cd data
    wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/


Environment Setup (uv)
---------------------

Create and install the environment:

    uv venv
    uv sync
    source .venv/bin/activate


What This Project Does
---------------------

- Loads raw EEG EDF files
- Extracts motor execution and motor imagery trials
- Maps signals to a 22-channel NeuroGPT montage
- Normalizes data and creates subject-independent splits
- Provides basic data exploration and visualization

Upcoming Classification Task
----------------------------

We perform a 4-class classification of motor-related EEG activity:
0 – Right hand imagined movement  
1 – Right hand real movement  
2 – Left hand imagined movement  
3 – Left hand real movement  

This task is interesting because it simultaneously tests lateralization
(left vs right motor cortex) and cognitive state (imagined vs executed),
making it a challenging and realistic benchmark for brain–computer
interface models such as NeuroGPT.