"""A timed-out CLI call must not leave descendants behind.

The frozen path killed only the direct child. `codex` is reached through a shim that
execs a node wrapper which spawns the native binary, so the process that matters is a
grandchild: it survived, kept its inherited pipes, and blocked forever. These tests
use a local stand-in that spawns its own child, so they exercise the mechanism without
any CLI, network or provider.
"""

import os
import subprocess
import sys
import time

import pytest

import vision


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _spawner(marker) -> str:
    """A program that starts a long-lived child, records its pid, then sleeps."""
    return (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c',\n"
        "    'import time, sys; sys.stdout.write(\"x\" * 100000); sys.stdout.flush(); time.sleep(60)'])\n"
        f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )


def test_a_timed_out_call_kills_the_grandchild_too(tmp_path):
    marker = tmp_path / "grandchild.pid"

    with pytest.raises(subprocess.TimeoutExpired):
        vision.run_cli([sys.executable, "-c", _spawner(marker)], timeout=3)

    grandchild = int(marker.read_text())
    for _ in range(20):                       # give the signal a moment to land
        if not _alive(grandchild):
            break
        time.sleep(0.1)
    assert not _alive(grandchild), (
        f"grandchild {grandchild} survived the timeout; this is the leak that filled the "
        f"machine during the JT run")


def test_a_normal_call_still_returns_its_output():
    done = vision.run_cli([sys.executable, "-c", "print('hello')"], timeout=30)

    assert done.returncode == 0
    assert done.stdout.strip() == "hello"


def test_input_is_still_delivered_on_stdin():
    done = vision.run_cli([sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"],
                          input="from stdin", timeout=30)

    assert done.stdout.strip() == "from stdin"


def test_a_timeout_reports_how_long_it_waited_and_what_was_captured(tmp_path):
    program = ("import sys, time\n"
               "sys.stderr.write('partial diagnostic line\\n')\n"
               "sys.stderr.flush()\n"
               "time.sleep(60)\n")

    with pytest.raises(subprocess.TimeoutExpired) as caught:
        vision.run_cli([sys.executable, "-c", program], timeout=2)

    assert caught.value.timeout == 2
    # Whatever the process managed to say before it was killed is preserved, so a
    # timeout is diagnosable rather than silent.
    captured = caught.value.stderr or ""
    assert "partial diagnostic line" in captured


def test_a_child_that_writes_more_than_the_pipe_buffer_is_still_killed(tmp_path):
    # The JT failure signature: the descendant blocked writing to a full pipe that
    # nobody was draining any more.
    program = ("import sys, time\n"
               "sys.stdout.write('x' * 500000)\n"
               "time.sleep(60)\n")
    started = time.time()

    with pytest.raises(subprocess.TimeoutExpired):
        vision.run_cli([sys.executable, "-c", program], timeout=3)

    assert time.time() - started < 20, "the call did not return promptly after its timeout"
