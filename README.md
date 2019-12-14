# Textbook Question Generation

This repository contains code for our CS 221 final project, *Automatic Quiz Question Generation from Textbooks*.  The code is built in PyTorch using elements from the OpenNMT framework.

Prerequisites include:
- pytorch==1.3.1
- torchtext
- opennmt-py
- pyyaml

Usage:

    python3 main.py yaml_args/copy_network.yaml

**Data for this project**, along with predicted questions, is available in `squad.tsv` and `tqa_transfer.tsv`.  Note that you may have to scroll right in `tqa_transfer.tsv` in order to view the questions.
