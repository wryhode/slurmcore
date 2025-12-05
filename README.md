# slurmcore tool
this is a simple(ish) tool to make slurmcore!

## issues
- popping is a prominent issue as the slices are spliced without any mixing. this is very fixable tho
- unfriendly parameter names

## usage
this project needs python 3.11 or 3.12 to work as some libraries do not support newer versions. i use python 3.11.

1. make a venv -> `py -3.11 -m venv ./venv`
2. activate the venv -> `venv/Scripts/activate`
3. install dependencies -> `pip install -r requirements.txt`
4. edit `main.py`, replace (at least) the first two arguments of `full_slurm` with your own stuff
5. run it -> `py -3.11 main.py`
6. listen to the output
7. make adjustments
8. go to step 4

## future development
features will be added whenever i feel like it. ill probably make this more user friendly as time goes on. a cli would be cool. maybe even a full gui but that's not happening any time soon.
