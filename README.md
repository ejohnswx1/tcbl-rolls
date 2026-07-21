# tcbl-rolls
This is a repository for all code related to the Francine (2024) hurricane boundary layer roll project.

## Package Requirements:

arm_pyart [docs](https://arm-doe.github.io/pyart/#)
pandas
numpy
xarray
matplotlib
scikit-learn
scikit-image
colormaps [docs](https://pratiman-91.github.io/colormaps/)
scipy

## Descriptions of Scripts

#### Functions.py
Script contains all functions responsible for automating the roll identification process. See utiliziations of each function in Main.ipynb.

#### Main.ipynb
The main automation example notebook, utilizing functions from Functions.py in order to to identify rolls for a sample time.

#### ComputeResidualVelocity.ipynb
The notebook that shows how to compure residual velocity given a quality-controlled radar file.
