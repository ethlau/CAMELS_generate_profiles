import subprocess
import os
import sys

# Path to your .so file
so_file = "/Users/ethlau/Research/CAMELS_fnth/cython_src/process_halos_cy.cpython-312-darwin.so"

# Check if file exists
if not os.path.exists(so_file):
    print(f"Error: {so_file} does not exist.")
    sys.exit(1)

# Check current RPATH entries
cmd = ["otool", "-l", so_file]
output = subprocess.check_output(cmd).decode('utf-8')
print("Before fixing:")
print(output)

# Path to remove (the duplicate one)
rpath = "/opt/homebrew/Caskroom/miniforge/base/lib"

# Use install_name_tool to remove all instances of this RPATH
while True:
    cmd = ["install_name_tool", "-delete_rpath", rpath, so_file]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # If command fails, we've removed all instances
            break
        print(f"Removed an instance of RPATH: {rpath}")
    except subprocess.CalledProcessError:
        break

# Add back a single instance of the RPATH
cmd = ["install_name_tool", "-add_rpath", rpath, so_file]
try:
    subprocess.check_call(cmd)
    print(f"Added a single instance of RPATH: {rpath}")
except subprocess.CalledProcessError as e:
    print(f"Failed to add RPATH: {e}")

# Check after fixing
cmd = ["otool", "-l", so_file]
output = subprocess.check_output(cmd).decode('utf-8')
print("After fixing:")
print(output)

print("Fix completed. Try importing the module now.")
