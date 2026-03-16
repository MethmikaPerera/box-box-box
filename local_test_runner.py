import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEST_CASES_DIR = ROOT / "data" / "test_cases" / "inputs"
EXPECTED_OUTPUTS_DIR = ROOT / "data" / "test_cases" / "expected_outputs"
RUN_COMMAND_FILE = ROOT / "solution" / "run_command.txt"


def load_run_command() -> str:
    if not RUN_COMMAND_FILE.exists():
        raise FileNotFoundError(
            f"Run command file not found: {RUN_COMMAND_FILE}\n"
            "Create solution/run_command.txt with e.g.:\n"
            "python solution/race_simulator.py"
        )
    return RUN_COMMAND_FILE.read_text(encoding="utf-8").strip()


def run_solution(command: str, input_json_path: Path):
    input_data = input_json_path.read_text(encoding="utf-8")

    completed = subprocess.run(
        command,
        input=input_data,
        text=True,
        capture_output=True,
        shell=True,
        cwd=ROOT,
    )
    return completed


def main():
    try:
        command = load_run_command()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    test_files = sorted(TEST_CASES_DIR.glob("test_*.json"))
    if not test_files:
        print(f"No test files found in {TEST_CASES_DIR}")
        sys.exit(1)

    has_answers = EXPECTED_OUTPUTS_DIR.exists()

    print("╔════════════════════════════════════════════════════════╗")
    print("║            Box Box Box - Local Test Runner             ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    print(f"Solution Command: {command}")
    print(f"Test Cases Found: {len(test_files)}")
    print()

    passed = 0
    failed = 0
    errors = 0

    print("Running tests...\n")

    for test_file in test_files:
        test_name = test_file.stem
        test_id = test_name.replace("test_", "TEST_")

        completed = run_solution(command, test_file)

        if completed.returncode != 0:
            print(f"✗ {test_id} - Execution error")
            if completed.stderr.strip():
                first_line = completed.stderr.strip().splitlines()[0]
                print(f"  Error: {first_line}")
            errors += 1
            continue

        raw_output = completed.stdout.strip()

        try:
            output_json = json.loads(raw_output)
        except Exception:
            print(f"✗ {test_id} - Invalid JSON output")
            failed += 1
            continue

        predicted = output_json.get("finishing_positions")
        race_id = output_json.get("race_id")

        if race_id is None or predicted is None:
            print(f"✗ {test_id} - Invalid output format")
            failed += 1
            continue

        if not isinstance(predicted, list) or len(predicted) != 20:
            print(f"✗ {test_id} - finishing_positions must contain exactly 20 drivers")
            failed += 1
            continue

        if len(set(predicted)) != 20:
            print(f"✗ {test_id} - Duplicate driver IDs in output")
            failed += 1
            continue

        if has_answers:
            answer_file = EXPECTED_OUTPUTS_DIR / f"{test_name}.json"
            if answer_file.exists():
                expected = json.loads(answer_file.read_text(encoding="utf-8")).get("finishing_positions")
                if predicted == expected:
                    print(f"✓ {test_id}")
                    passed += 1
                else:
                    print(f"✗ {test_id} - Incorrect prediction")
                    failed += 1
            else:
                print(f"? {test_id} - Output generated (no answer file found)")
                passed += 1
        else:
            print(f"? {test_id} - Output generated (no answer key to verify)")
            passed += 1

    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║                        Results                         ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    print(f"Total Tests:    {len(test_files)}")
    print(f"Passed:         {passed}")
    print(f"Failed:         {failed}")
    if errors:
        print(f"Errors:         {errors}")
    print()

    pass_rate = (passed * 100.0 / len(test_files)) if test_files else 0.0
    print(f"Pass Rate:      {pass_rate:.1f}%")
    print()

    if passed == len(test_files):
        print("🏆 Perfect score! All tests passed!")
        sys.exit(0)
    elif passed > 0:
        print("Keep improving! Check failed test cases.")
        sys.exit(0)
    else:
        print("No tests passed. Review your implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()