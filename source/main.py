from opponent_policy_runtime import apply_runtime_patch

# Install the 3.9.3 opponent-policy + survival-selector runtime before the GUI
# creates the card-game bot.  Fail closed: a missing model/patch must never
# silently launch the older production selector.
if not apply_runtime_patch():
    raise RuntimeError("HeroesWM 3.9.3 runtime patch failed to load")

from gui import run_gui


if __name__ == "__main__":
    run_gui()
