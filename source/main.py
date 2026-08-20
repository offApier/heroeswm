from opponent_policy_runtime import apply_runtime_patch

# Install the learned population opponent policy before the GUI creates the
# card-game bot.  Failure to load the optional model is fail-safe: the existing
# 3.9.1 strategy remains untouched.
apply_runtime_patch()

from gui import run_gui


if __name__ == "__main__":
    run_gui()
