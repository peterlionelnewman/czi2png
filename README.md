# czi2png
This wrapper script (around the aicslibczi library) converts your '.czi' files to '.png's

## Info
Written by: Pete, 2023, (peterlionelnewman @ gmail com / p.newman @ sydney edu au)
Helpful for students I worked with

1. searches a folder for .czi files
2. exports a mip of each czi channel

## To run, something like:
  conda create -n czi2png python=3.10.6
  conda activate czi2png
  pip install -r requirements.txt

## Updates TBD
- best ran from a conda venv
- see requirements.txt

## Notes
- Go check out the Allen Institue of Cell Science package !!!
- This exists because sometimes the AICS package is buggy with rounding errors. But otherwise we AOK.

