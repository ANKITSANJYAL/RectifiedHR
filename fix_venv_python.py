#!/usr/bin/env python3
"""
fix_venv_python.py - Fix Python interpreter in virtual environment for cluster nodes

This script fixes the issue where compute nodes have different Python versions
by ensuring the venv uses the correct Python 3 interpreter.
"""

import os
import sys
import shutil
from pathlib import Path

def find_python3():
    """Find the best Python 3 interpreter available."""
    # Common paths for Python 3
    candidates = [
        '/opt/anaconda3/bin/python3',
        '/opt/anaconda3/bin/python',
        '/usr/bin/python3',
        '/usr/local/bin/python3',
        shutil.which('python3'),
        shutil.which('python')
    ]
    
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            try:
                # Test if it's Python 3.8+
                import subprocess
                result = subprocess.run([candidate, '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip().split()[1]
                    major, minor = map(int, version.split('.')[:2])
                    if major == 3 and minor >= 8:
                        print(f"✅ Found suitable Python: {candidate} (version {version})")
                        return candidate
            except Exception:
                continue
    return None

def fix_venv_python(venv_path="venv"):
    """Fix the Python interpreter in the virtual environment."""
    venv_path = Path(venv_path)
    if not venv_path.exists():
        print(f"❌ Virtual environment not found at {venv_path}")
        return False
    
    # Find the correct Python 3
    python3_path = find_python3()
    if not python3_path:
        print("❌ No suitable Python 3 found!")
        return False
    
    # Fix the python symlink in venv/bin/
    bin_path = venv_path / "bin"
    python_link = bin_path / "python"
    
    print(f"🔧 Fixing Python interpreter in {venv_path}")
    
    # Backup existing python link
    if python_link.exists() or python_link.is_symlink():
        backup_path = bin_path / "python.backup"
        if backup_path.exists():
            backup_path.unlink()
        python_link.unlink()  # Remove the existing symlink first
        print(f"🗑️  Removed existing python symlink")
    
    # Also fix python3 symlink if it exists
    python3_link = bin_path / "python3"
    if python3_link.exists() or python3_link.is_symlink():
        backup_path = bin_path / "python3.backup"
        if backup_path.exists():
            backup_path.unlink()
        python3_link.unlink()
        print(f"�️  Removed existing python3 symlink")
    
    # Create new symlinks
    python_link.symlink_to(python3_path)
    python3_link.symlink_to(python3_path)
    print(f"🔗 Created new symlinks: python -> {python3_path}")
    print(f"🔗 Created new symlinks: python3 -> {python3_path}")
    
    # Test the fix
    try:
        import subprocess
        result = subprocess.run([str(python_link), '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Fixed Python interpreter: {version}")
            return True
    except Exception as e:
        print(f"❌ Failed to test fixed Python: {e}")
    
    return False

if __name__ == "__main__":
    print("🔧 Python Virtual Environment Fixer")
    print("=" * 40)
    
    venv_path = sys.argv[1] if len(sys.argv) > 1 else "venv"
    success = fix_venv_python(venv_path)
    
    if success:
        print("\n🎉 Virtual environment fixed successfully!")
        print(f"💡 Now activate with: source {venv_path}/bin/activate")
    else:
        print("\n❌ Failed to fix virtual environment!")
        sys.exit(1)
