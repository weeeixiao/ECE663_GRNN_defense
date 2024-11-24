## Running Environment
The environment is available in `env/fl_proj_env.yml`. I currently use **Conda** to manage the environment.

## Commands
The entrance to our implementation is `defense_trial.py`. Using `python defense_trial.py` is sufficient to begin and I have tested on *Ubuntu 20.08*. To customize the arguments, please refer to `utils/options.py` for further information.

## Current Progress
- The project has so far fulfilled the original design target. To be specific, we aim to defend GRNN attack, and our DP implementation indeed achieve the goal, with reasonable accuracy loss.
- I didn't modify the structure of GRNN, and there should be consequent work on refining it. In future I can look into some more advanced framework.
- The current implementation of DP is not audited.
