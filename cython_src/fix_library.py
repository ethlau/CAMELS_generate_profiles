# fix_library.py
import subprocess
import os

so_file = "process_halos_cy.cpython-312-darwin.so"

# Check current RPATH entries
cmd = ["otool", "-l", so_file]
output = subprocess.check_output(cmd).decode('utf-8')
print("Before fixing:")
print(output)

# Remove duplicate RPATHs
rpath = "/opt/homebrew/Caskroom/miniforge/base/lib"
cmd = ["install_name_tool", "-delete_rpath", rpath, so_file]
try:
    subprocess.check_call(cmd)
    print(f"Removed RPATH: {rpath}")
except subprocess.CalledProcessError:
    print(f"Failed to remove RPATH: {rpath}")

# Check after fixing
cmd = ["otool", "-l", so_file]
output = subprocess.check_output(cmd).decode('utf-8')
print("After fixing:")
print(output)
