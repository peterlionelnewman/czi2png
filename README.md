# czi2png

This wrapper GUI + script (around the aicslibczi library) for batch processing / converting your '.czi' files to '.png's

## Info

- Written by: Pete, 2022, (peterlionelnewman)
- Paper doi: doi.org/10.1002/advs.202204741
- Paper url: https://onlinelibrary.wiley.com/doi/10.1002/advs.202204741
- Cite us if you use any part of this script in anyway :-)

1. searches a folder for .czi files
2. exports a mip of each czi channel

## To run, use something something like:
```
    conda create -n czi2png python=3.10.6 pip
    conda activate czi2png
    cd <path to czi2png>
    pip install -r requirements.txt
    python main.py 
```

- best ran from a conda venv
- see requirements.txt

## To use

1. Click and specify the search path button search fo czi's
2. Click and specify the saving path
3. Select options for png export
4. Click 'Converyt 2 png'!

i.e.:
![example.png](https://github.com/peterlionelnewman/czi2png/blob/main/example.png)


## Notes

- Go check out the Allen Institue of Cell Science package !!!
- This exists because sometimes the AICS package is buggy with rounding errors. But otherwise we AOK.
