#!/usr/bin/env python3
"""The ROCm plugin's release build and the device rules must name the same architectures: a card devices.cpp calls usable must have code
objects in the published plugin (or the first query fails inside HIP), and a built architecture devices.cpp does not list is refused for
nothing. Compares AMDGPU_TARGETS in .github/workflows/plugins.yml with GgmlRocmPluginArchs() in src/devices.cpp."""
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parent.parent
workflow = (root / ".github/workflows/plugins.yml").read_text()
devices = (root / "src/devices.cpp").read_text()
built = re.search(r'-DAMDGPU_TARGETS=([^"\s]+)', workflow)
listed = re.search(r"GgmlRocmPluginArchs\(\) \{.*?\{(.*?)\};", devices, re.S)
if not built or not listed:
    sys.exit("check_rocm_archs: could not find AMDGPU_TARGETS in plugins.yml or the list in GgmlRocmPluginArchs()")
built_set = set(built.group(1).split(";"))
listed_set = set(re.findall(r'"(gfx\w+)"', listed.group(1)))
if built_set != listed_set:
    print("check_rocm_archs: the ROCm plugin is built for", sorted(built_set), "but devices.cpp calls usable", sorted(listed_set))
    print("  only built:", sorted(built_set - listed_set), " only listed:", sorted(listed_set - built_set))
    sys.exit(1)
print("check_rocm_archs: OK,", len(built_set), "architectures")
