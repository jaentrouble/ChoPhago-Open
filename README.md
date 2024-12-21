# ChoPhago-Open
## Intro
ChoPhago is an AI based helper program for the game LostArk's transcendence system.
Now that transcendence has changed to a simpler monopoly-like system, ChoPhago is no longer needed.

**This repository is for archival purposes only.**

## Files

### train.py
- Use 'configs/config.json' to change the training parameters.
- If you are using **tensorflow > 2.15**, you need to change the wrapper model to use keras Layer instead of raw tensorflow ops.
    - You can do this by uncommenting the code in the 'train.py' file.
- Board configurations are in configs/board_info.json
- The results will be saved in 'logs/{name}' directory.
    - I have left a sample log file of weapon lv.6 in the 'logs' directory.
    - **The checkpoint of the sample log only works with exactly the same environment in 'requirements.txt'.**
    - You can still run train.py file with other tensorflow versions, but the sample checkpoint may not work.

### eval_result.ipynb
- Use this file to evaluate and visualize the training results.

### requirements.txt
- The requirements are
    - tensorflow
    - tqdm
    - numpy
    - numba
    - keras-nlp (now changed to keras-hub; I used the old version)
    - psutil
    - matplotlib
- Tensorflow tends to update frequently and not backward-compatible, so the codes may not work in future versions.

### configs/config.json
- The training parameters are in this file.

## Disclaimer
The Transcendence system, LostArk, and all related trademarks and logos are the property of Smilegate RPG. I do not own any of the trademarks and logos, nor am I affiliated with Smilegate RPG.