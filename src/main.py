# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import time

from config import CFG

if __name__ == "__main__":
    # Store results
    timings = {}

    # List of ancilla modes to test
    ancilla_modes = ["pass", "project", "trace_out"]

    for mode in ancilla_modes:
        print(f"\nRunning with ancilla_mode = '{mode}'...")
        # Set the mode in config
        CFG.ancilla_mode = mode
        # Reload training module to ensure config is picked up
        importlib.invalidate_caches()
        import training

        start = time.time()
        training.t = training.Training()
        training.t.run()
        elapsed = time.time() - start
        timings[mode] = elapsed
        print(f"Time for ancilla_mode '{mode}': {elapsed:.2f} seconds")

    print("\nSummary of timings:")
    for mode in ancilla_modes:
        print(f"{mode:>10}: {timings[mode]:.2f} seconds")
