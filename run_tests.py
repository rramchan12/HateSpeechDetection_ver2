#!/usr/bin/env python
"""Test runner script for the ToxiGen Hate Speech Detection project.

This script provides convenient commands for running different types of tests.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n🧪 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    return subprocess.call(cmd)


def main():
    """Main test runner."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [command]")
        print("\nAvailable commands:")
        print("  all        - Run all tests with coverage")
        print("  unit       - Run only unit tests")
        print("  integration - Run only integration tests") 
        print("  data       - Run data-dependent tests (with output)")
        print("  fast       - Run tests without coverage")
        print("  coverage   - Generate coverage report")
        return 1

    command = sys.argv[1].lower()
    
    base_cmd = [sys.executable, "-m", "pytest", "tests/"]
    
    if command == "all":
        cmd = base_cmd + ["-v", "--cov-fail-under=70"]
        return run_command(cmd, "Running all tests with coverage")
    
    elif command == "unit":
        cmd = base_cmd + ["-v", "-m", "unit", "--cov-fail-under=0"]
        return run_command(cmd, "Running unit tests only")
    
    elif command == "integration":
        cmd = base_cmd + ["-v", "-m", "integration", "--cov-fail-under=0"]
        return run_command(cmd, "Running integration tests only")
        
    elif command == "data":
        cmd = base_cmd + ["-v", "-m", "data", "-s", "--cov-fail-under=0"]
        return run_command(cmd, "Running data tests with output")
        
    elif command == "fast":
        cmd = base_cmd + ["-v", "--no-cov"]
        return run_command(cmd, "Running tests without coverage")
        
    elif command == "coverage":
        cmd = [sys.executable, "-m", "pytest", "tests/", "--cov-report=html", "--cov-report=term"]
        return run_command(cmd, "Generating coverage report")
        
    else:
        print(f"Unknown command: {command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())