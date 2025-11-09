import sys
import subprocess
from pathlib import Path
import textwrap

SCRIPT = Path(__file__).resolve().parents[1] / "sort_two_rows.py"

def run_script(tmp_path, extra_args, input_text):
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    input_file.write_text(input_text, encoding="utf-8")

    cmd = [sys.executable, str(SCRIPT), "-i", str(input_file), "-o", str(output_file)] + extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc, output_file

def read_output(output_path):
    if not output_path.exists():
        return None
    return output_path.read_text(encoding="utf-8")

def test_numeric_comma_separated(tmp_path):
    input_text = "3, 1, 2\n9, 7, 8\n"
    proc, out = run_script(tmp_path, [], input_text)
    assert proc.returncode == 0
    content = read_output(out).splitlines()
    assert content == ["1, 2, 3", "7, 8, 9"]

def test_string_space_separated(tmp_path):
    input_text = "a z b\nc d e\n"
    proc, out = run_script(tmp_path, [], input_text)
    assert proc.returncode == 0
    content = read_output(out).splitlines()
    assert content == ["a, b, z", "c, d, e"]

def test_combine_flag(tmp_path):
    input_text = "2 1\n3 4\n"
    proc, out = run_script(tmp_path, ["--combine"], input_text)
    assert proc.returncode == 0
    content = read_output(out).splitlines()
    assert content == ["1, 2, 3, 4"]

def test_reverse_flag(tmp_path):
    input_text = "10,2,3\n5,1\n"
    proc, out = run_script(tmp_path, ["--reverse"], input_text)
    assert proc.returncode == 0
    content = read_output(out).splitlines()
    assert content == ["10, 3, 2", "5, 1"]

def test_force_numeric_with_non_numeric_row(tmp_path):
    input_text = "a, b\n1,2\n"
    # forcing numeric when first row has
