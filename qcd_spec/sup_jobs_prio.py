"""Second runner on the supervisor-spec queue: the 30 priority jobs of sup_jobs.py (k = 8 + raw reference, the
LayerNorm ablation, the code-size sweep on seed 0) in REVERSE order, so it meets the first runner in the middle.
Duplicates are prevented by the run lock in spec_run.py."""
import subprocess
import sys
from pathlib import Path

lines = subprocess.run([sys.executable, str(Path(__file__).with_name('sup_jobs.py'))], capture_output=True,
                       text=True, check=True).stdout.splitlines()
prio = [l for l in lines if ('_s1_' not in l and '_s2_' not in l) or ' --k 8 ' in l or 'mog_raw' in l]
print('\n'.join(reversed(prio)))
