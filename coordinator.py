"""
coordinator.py — Orchestrates the full training + validation loop.
Run this once; it will keep retraining until all GT flags are detected.
"""
import subprocess
import sys
import os

WORKSPACE = r'C:\Users\mido\Documents\antigravity\focused-babbage'
MAX_ROUNDS = 5

def run(cmd, **kwargs):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=WORKSPACE, **kwargs)
    return result

def main():
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'#'*60}")
        print(f"#  ROUND {round_num} of {MAX_ROUNDS}")
        print(f"{'#'*60}")

        # Step 1 — Train
        print("\n--- Training ---")
        run(["python", "train_flag_yolo.py"], check=True)

        # Step 2 — Validate (tiled @ 320)
        print("\n--- Validation (tiled) ---")
        res = run(
            ["python", "validate_tiled.py"],
            capture_output=False,
            text=True
        )

        if res.returncode == 0:
            # Check stdout for success
            pass

        # Read the last validation result from stdout
        val_result = subprocess.run(
            ["python", "validate_tiled.py"],
            cwd=WORKSPACE, capture_output=True, text=True
        )
        print(val_result.stdout)

        if "ALL CRITICAL FLAGS DETECTED CORRECTLY" in val_result.stdout:
            print(f"\n🏆 SUCCESS on round {round_num}!")
            sys.exit(0)
        else:
            print(f"\n❌ Round {round_num} failed. Adjusting and retrying...")

            # Augment the dataset slightly on retries — add more negative samples
            # by adjusting blur and scale for harder flags
            if round_num < MAX_ROUNDS:
                print("Regenerating dataset with harder augmentation...")
                run([
                    "python", "-c",
                    "import generate_synthetic_dataset; generate_synthetic_dataset.generate_dataset(num_train=3200, num_val=800)"
                ], check=True)

    print(f"\n❌ Max rounds ({MAX_ROUNDS}) reached without success.")
    sys.exit(1)

if __name__ == '__main__':
    main()
