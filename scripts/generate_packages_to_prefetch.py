"""Generate list of packages to be prefetched in Cachi2 and used in Konflux for hermetic build.

This script performs several steps:
    1. removes torch+cpu dependency from project file
    2. generates requirements.txt file from pyproject.toml + pdm.lock
    3. removes all torch dependencies (including CUDA/Nvidia packages)
    4. downloads torch+cpu wheel
    5. computes hashes for this wheel
    6. adds the URL to wheel + hash to resulting requirements.txt file
"""

import shutil
import subprocess
import tempfile
from os.path import join
from urllib.request import urlretrieve

# just these files are needed as project stub, no other configs and/or sources are needed
PROJECT_FILES = ("pyproject.toml", "pdm.lock", "LICENSE", "README.md")

# registry with Torch wheels (CPU variant)
TORCH_REGISTRY = "https://download.pytorch.org/whl/cpu"
TORCH_VERSION = "2.2.2"
TORCH_WHEEL = f"torch-{TORCH_VERSION}%2Bcpu-cp311-cp311-linux_x86_64.whl"


def shell(command, directory):
    """Run command via shell inside specified directory."""
    return subprocess.check_output(command, cwd=directory, shell=True)  # noqa S602


def copy_project_stub(directory):
    """Copy all files that represent project stub into specified directory."""
    for project_file in PROJECT_FILES:
        shutil.copy(project_file, directory)


def remove_torch_dependency(directory):
    """Remove torch (specifically torch+cpu) dependency from the project.toml file."""
    shell("pdm remove torch", directory)


def generate_requirements_file(work_directory):
    """Generate file requirements.txt that contains hashes for all packages."""
    shell("pip-compile -vv pyproject.toml --generate-hashes", work_directory)


def remove_package(directory, source, target, package_prefix):
    """Remove package or packages with specified prefix from the requirements file."""
    package_block = False

    with open(join(directory, source)) as fin:
        with open(join(directory, target), "w") as fout:
            for line in fin:
                if line.startswith(package_prefix):
                    print(line)
                    package_block = True
                elif package_block:
                    # the whole block with hashes needs to be filtered out
                    if not line.startswith("    "):
                        # end of package block detected
                        package_block = False
                if not package_block:
                    fout.write(line)


def remove_unwanted_dependencies(directory):
    """Remove all unwanted dependencies from requirements file, creating in-between files."""
    # the torch itself
    remove_package(directory, "requirements.txt", "step1.txt", "torch")

    # all CUDA-related packages (torch depends on them)
    remove_package(directory, "step1.txt", "step2.txt", "nvidia")


def wheel_url(registry, wheel):
    """Construct full URL to wheel."""
    return f"{registry}/{wheel}"


def download_wheel(directory, registry, wheel):
    """Download selected wheel from registry."""
    url = wheel_url(registry, wheel)
    into = join(directory, wheel)
    urlretrieve(url, into)  # noqa S310


def generate_hash(directory, registry, wheel, target):
    """Generate hash entry for given wheel."""
    output = shell(f"pip hash {wheel}", directory)
    hash_line = output.decode("ascii").splitlines()[1]
    with open(join(directory, target), "w") as fout:
        url = wheel_url(registry, wheel)
        fout.write(f"torch @ {url} \\\n")
        fout.write(f"    {hash_line}\n")


def generate_list_of_packages():
    """Generate list of packages, take care of unwanted packages and wheel with Torch package."""
    work_directory = tempfile.mkdtemp()
    print(f"Work directory {work_directory}")

    copy_project_stub(work_directory)
    remove_torch_dependency(work_directory)
    generate_requirements_file(work_directory)

    remove_unwanted_dependencies(work_directory)
    download_wheel(work_directory, TORCH_REGISTRY, TORCH_WHEEL)
    shutil.copy(join(work_directory, "step2.txt"), "requirements.txt")

    generate_hash(work_directory, TORCH_REGISTRY, TORCH_WHEEL, "hash.txt")
    shell("cat step2.txt hash.txt > step3.txt", work_directory)
    shutil.copy(join(work_directory, "step3.txt"), "requirements.txt")

    # optional cleanup step
    # (for now it might be better to see 'steps' files to check if everything's ok
    # shutil.rmtree(work_directory)


if __name__ == "__main__":
    generate_list_of_packages()
