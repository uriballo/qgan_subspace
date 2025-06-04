import importlib
import time

import config as cf

if __name__ == "__main__":
    # Store results
    timings = {}

    # List of ancilla modes to test
    ancilla_modes = ["pass", "project", "trace_out"]

    for mode in ancilla_modes:
        print(f"\nRunning with ancilla_mode = '{mode}'...")
        # Set the mode in config
        cf.ancilla_mode = mode
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
