# pytest-based unit tests for check_operator_spacing.py
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "check_operator_spacing.py"

def run_on_contents(contents: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        f = td_path / "test.cc"
        f.write_text(contents, encoding="utf-8")
        # We need a git repo so git ls-files works; initialize minimal repo
        # and add the file so it appears in git ls-files
        subprocess.run(["git", "init"], cwd=td, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=td, check=True)
        subprocess.run(["git", "config", "user.name", "tester"], cwd=td, check=True)
        subprocess.run(["git", "add", str(f.name)], cwd=td, check=True)
        subprocess.run(["git", "commit", "-m", "add"], cwd=td, check=True, stdout=subprocess.PIPE)
        # copy the script into the temp dir so it can be executed
        script_dst = td_path / SCRIPT.name
        script_dst.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(script_dst)], cwd=td, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc

def test_allowed_no_spaces():
    code = "void f() { bool b = (x<lower); }\n"
    proc = run_on_contents(code)
    assert proc.returncode == 0, proc.stdout + proc.stderr

def test_allowed_both_sides_spaces():
    code = "void f() { bool b = (x < lower); }\n"
    proc = run_on_contents(code)
    assert proc.returncode == 0, proc.stdout + proc.stderr

def test_disallow_mixed_right_space():
    code = "void f() { bool b = (x< lower); }\n"
    proc = run_on_contents(code)
    assert proc.returncode != 0
    assert "Inconsistent spacing" in proc.stdout

def test_disallow_mixed_left_space():
    code = "void f() { bool b = (x <lower); }\n"
    proc = run_on_contents(code)
    assert proc.returncode != 0
    assert "Inconsistent spacing" in proc.stdout

def test_composite_expression_matches_issue():
    code = "PrimExpr out_range = Or(x<lower, x> upper);\n"
    proc = run_on_contents(code)
    assert proc.returncode != 0
    # should flag the x> upper instance
    assert "Inconsistent spacing" in proc.stdout