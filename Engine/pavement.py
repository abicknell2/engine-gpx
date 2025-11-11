from collections import defaultdict
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import re
from shutil import copy, copy2, copytree, move, rmtree, which
import subprocess
import sys
import tempfile
from typing import Optional

from paver.easy import Bunch, cmdopts, options, pushd, sh, task

# Get the directory where the pavement.py file is located
base_dir: str = os.path.abspath(os.path.dirname(__file__))
LLX_DIR = Path("./test/data/llx_integration_tests")
JSON_DIR = Path("./test/data/json_data")

options(
    mypy=Bunch(output_file="mypy_results.txt"),
    flake8=Bunch(output_file="flake8_results.txt"),
    pylint=Bunch(output_file="pylint_results.txt"),
)


def project_and_engine_roots():
    """Return (project_root, engine_root) as Path objects."""
    engine_root = Path(__file__).resolve().parent
    project_root = engine_root.parent
    return project_root, engine_root


def remove_file(file_path: str) -> None:
    """Delete a file if it exists to ensure fresh results."""
    if os.path.exists(file_path):
        os.remove(file_path)


def compare_results(current_file: str, saved_file: str) -> tuple[Optional[list[str]], Optional[list[str]]]:
    """Compare the current run with the saved results and return new and removed errors."""
    if not os.path.exists(saved_file):
        return None, None  # No saved results, all errors are new

    with open(saved_file, "r") as f:
        saved_errors: set[str] = set(f.readlines())

    with open(current_file, "r") as f:
        current_errors: set[str] = set(f.readlines())

    new_errors: set[str] = current_errors - saved_errors  # Find only new errors
    removed_errors: set[str] = saved_errors - current_errors  # Find removed errors

    return list(new_errors), list(removed_errors)


@task
def isort() -> None:
    """Run isort to sort imports."""
    sh(f"isort {base_dir} .")


@task
def add_trailing_commas() -> None:
    """Add trailing commas to all Python files."""
    sh("add-trailing-comma $(git ls-files '*.py')")


@task
def yapf() -> None:
    """Run yapf to format code."""
    sh(f"yapf --style .style.yapf --in-place --recursive {base_dir} .")


@task
def prettier() -> None:
    """Format JSON and MD files using Prettier"""
    sh("npx prettier --write '**/*.json'")
    sh("npx prettier --write '**/*.md'")


@task
def pylint() -> None:
    """Run pylint, compare results, and show detailed messages."""
    output_file: str = "pylint_current.txt"
    saved_file: str = options.pylint.output_file

    remove_file(output_file)
    sh(f"pylint --rcfile={base_dir}/.pylintrc $(git ls-files '*.py') --output-format=text > {output_file} || true")

    new_errors, removed_errors = compare_results(output_file, saved_file)

    if new_errors and any(line.strip() for line in new_errors):
        print(f"\n⚠️  {len(new_errors)} new Pylint issues found! 🚨")
        print("".join(new_errors))
    else:
        print("\n✅ No new Pylint issues found! 🎉")

    if removed_errors and any(line.strip() for line in removed_errors):
        print(f"\n✅ {len(removed_errors)} Pylint issues have been fixed! 🎉")

    remove_file(output_file)


@task
def pylint_save() -> None:
    """Run pylint and save results."""
    output_file: str = options.pylint.output_file
    remove_file(output_file)
    sh(f"pylint --rcfile={base_dir}/.pylintrc $(git ls-files '*.py') --output-format=text > {output_file} || true")
    print("\n✅ Pylint results have been updated and saved to 'pylint_results.txt'.")


@task
def flake8() -> None:
    """Run flake8, compare results, and show detailed messages."""
    output_file: str = "flake8_current.txt"
    saved_file: str = options.flake8.output_file

    remove_file(output_file)
    sh(f"flake8 {base_dir} --format=default --color=never --output-file={output_file} || true")

    new_errors, removed_errors = compare_results(output_file, saved_file)

    if new_errors and any(line.strip() for line in new_errors):
        print(f"\n⚠️  {len(new_errors)} new Flake8 issues found! 🚨")
        print("".join(new_errors))
    else:
        print("\n✅ No new Flake8 issues found! 🎉")

    if removed_errors and any(line.strip() for line in removed_errors):
        print(f"\n✅ {len(removed_errors)} Flake8 issues have been fixed! 🎉")

    remove_file(output_file)


@task
def flake8_save() -> None:
    """Run flake8 and save results."""
    output_file: str = options.flake8.output_file
    remove_file(output_file)
    sh(f"flake8 {base_dir} --format=default --color=never --output-file={output_file} || true")
    print("\n✅ Flake8 results have been updated and saved to 'flake8_results.txt'.")


@task
def mypy() -> None:
    """Run mypy, compare results, and show detailed messages."""
    output_file: str = "mypy_current.txt"
    saved_file: str = options.mypy.output_file

    remove_file(output_file)

    # Run mypy and store full results
    sh(f"mypy . --no-incremental --show-traceback --show-error-context --show-column-numbers > {output_file} || true")

    # Read the full mypy output
    with open(output_file, "r") as f:
        error_output = f.readlines()

    # Group errors by file and sort by line number
    grouped_errors = group_mypy_errors(error_output)

    print("\n📢 Full mypy output (Grouped by file):")
    print_grouped_errors(grouped_errors)

    # Compare against saved results
    new_errors, removed_errors = compare_results(output_file, saved_file)

    if new_errors:
        print(f"\n⚠️  {len(new_errors)} new Mypy issues found! 🚨")
        grouped_new_errors = group_mypy_errors(new_errors)
        print_grouped_errors(grouped_new_errors)

    else:
        print("\n✅ No new Mypy issues found! 🎉")

    if removed_errors:
        print(f"\n✅ {len(removed_errors)} Mypy issues have been fixed! 🎉")

    remove_file(output_file)


def group_mypy_errors(errors: list[str]) -> dict[str, list[str]]:
    """
    Groups mypy errors by file and sorts them by line number.

    Returns:
        A dictionary where keys are file paths and values are lists of sorted error messages.
    """
    error_dict = defaultdict(list)

    for error in errors:
        parts = error.split(":", 3)  # Expected format: filename:line:column: error message
        if len(parts) >= 3 and parts[1].isdigit():
            file_path = parts[0]
            error_dict[file_path].append(error.strip())

    # Sort errors within each file by line number
    for file_path in error_dict:
        error_dict[file_path].sort(key=lambda x: int(x.split(":")[1]))

    return error_dict


def print_grouped_errors(grouped_errors: dict[str, list[str]]) -> None:
    """
    Prints grouped mypy errors in a structured format.
    """
    for file, errors in sorted(grouped_errors.items()):
        print(f"\n📌 {file}:")
        for error in errors:
            print(f"  {error}")  # Indented for readability


@task
def mypy_save() -> None:
    """Run mypy and save results."""
    output_file: str = options.mypy.output_file
    remove_file(output_file)
    sh(f"mypy . --no-incremental --show-traceback --show-error-context --show-column-numbers > {output_file} || true")
    print("\n✅ Mypy results have been updated and saved to 'mypy_results.txt'.")


@task
def bandit() -> None:
    """Run Bandit and display the results in the terminal."""
    sh("bandit -r . -c bandit.yml")


@task
def bandit_save() -> None:
    """Run Bandit and save results to bandit_results.txt."""
    output_file: str = "bandit_results.txt"
    remove_file(output_file)
    sh(f"bandit -r . -c bandit.yml > {output_file} || true")
    print("\n✅ Bandit results have been updated and saved to 'bandit_results.txt'.")


@task
def process_llx() -> None:
    """
    Paver task that checks if .llx files exist in json_data, copies them if not,
    converts to JSON, extracts resultsIndex, and stores it separately.
    """
    # Ensure JSON data directory exists
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    for llx_file in LLX_DIR.glob("*.llx"):  # Iterate over all .llx files
        json_subfolder = JSON_DIR / llx_file.stem
        json_file = json_subfolder / f"{llx_file.stem}.json"
        results_file = json_subfolder / f"{llx_file.stem}_results.json"

        # Check if JSON equivalent already exists
        if json_subfolder.exists():
            print(f"Skipping {llx_file.name}, already converted.")
            continue

        # Create subfolder and copy file
        json_subfolder.mkdir()
        copy(llx_file, json_file)

        # Read and modify the new JSON file
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)  # Load JSON data
            except json.JSONDecodeError:
                print(f"Error: {llx_file.name} is not valid JSON. Skipping.")
                continue

        # Extract resultsIndex if it exists
        if "results" in data:
            results_data = data.pop("results")

            # Save resultsIndex as a separate JSON file
            with open(results_file, "w", encoding="utf-8") as rf:
                json.dump(results_data, rf, indent=4)
                print(f"Extracted resultsIndex to {results_file}")

        else:
            print(f"No resultsIndex found in {json_file}, file unchanged.")

        print(f"Processed {llx_file.name} -> {json_file}")


@task
def lint() -> None:
    """Run all linting tasks and print a summary at the end."""
    print("\n🔍 Running Linting Tasks...\n")
    pylint()
    isort()
    process_llx()
    prettier()
    flake8()
    mypy()
    yapf()
    bandit()
    print("\n✅ Linting completed!")


@task
def lint_save() -> None:
    """Run all linting tasks and save results."""
    print("\n🔍 Running Linting Tasks and Saving Results...\n")
    pylint_save()
    isort()
    process_llx()
    prettier()
    flake8_save()
    mypy_save()
    yapf()
    bandit_save()
    print("\n✅ Linting completed and results saved!")


@task
def coverage() -> None:
    """Run test suite with coverage tracking."""
    sh("coverage run --source=. --branch -m pytest")


@task
def coverage_report() -> None:
    """Generate coverage report in the terminal."""
    sh("coverage report -m")


@task
def coverage_html() -> None:
    """Generate coverage report as HTML."""
    sh("coverage html")


@task
def coverage_clean() -> None:
    """Clear previous coverage data."""
    sh("coverage erase")


@task
def coverage_all() -> None:
    """Run full coverage workflow: clean, run, report, and HTML."""
    coverage_clean()
    coverage()
    coverage_report()
    coverage_html()
    print("✅ Coverage workflow complete!")


options(gpx=Bunch(path=None), )


def get_gpx_path(repo_root: str) -> Optional[Path]:
    default_GPX = (repo_root / "GPX").resolve()
    default_gpx = (repo_root / "gpx").resolve()

    # Ask the user where GPX lives
    print("Where is the GPX package directory?")
    print(f"  [1] {default_GPX}  (default)")
    print(f"  [2] {default_gpx}")
    print("  [3] Enter a custom path")

    choice = (input("Choose 1 / 2 / 3 [1]: ").strip() or "1")

    if choice == "1":
        gpx_root = default_GPX
    elif choice == "2":
        gpx_root = default_gpx
    elif choice == "3":
        path_in = input("Path to GPX directory: ").strip()
        gpx_root = Path(path_in).expanduser().resolve()
    else:
        raise SystemExit("✖ Invalid selection—aborting.")

    options.gpx.path = gpx_root
    return gpx_root


@task
def build_gpx():
    """
    Build a GPX source-distribution in the chosen GPX folder
    and copy/overwrite it into Code/Engine as gpx-zipped.tar.gz.
    """

    engine_root = Path(__file__).resolve().parent
    repo_root = engine_root.parent

    gpx_root = get_gpx_path(repo_root)

    if not gpx_root.exists():
        raise SystemExit(f"✖ GPX directory not found: {gpx_root}")

    # Build the sdist in that folder
    with pushd(str(gpx_root)):
        sh("python -m build --sdist --outdir .")  # gpx-<ver>.tar.gz

    # Locate newest artefact
    try:
        newest = max(gpx_root.glob("gpx-*.tar.gz"), key=lambda p: p.stat().st_mtime)
    except ValueError:
        raise SystemExit("✖ No gpx-*.tar.gz produced—check build output.")

    # Overwrite copy in Engine
    dest = engine_root / "gpx-zipped.tar.gz"
    dest.unlink(missing_ok=True)  # remove existing file if present
    move(str(newest), dest)  # rename/move

    print(f"✓ Created {dest.relative_to(repo_root)}")


@task
def build_ad():
    """
    Build a wheel for the installed 'ad' package and copy it into the Engine directory.
    """
    engine_root = Path(__file__).resolve().parent

    # locate the installed package folder  (.../site-packages/ad)
    spec = importlib.util.find_spec("ad")
    if spec is None or spec.origin is None:
        sys.exit("✖ 'ad' is not installed in the active virtual-environment.")
    pkg_dir = Path(spec.origin).parent
    site_pkgs = pkg_dir.parent

    # locate dist-info folder
    dist_info_dirs = list(site_pkgs.glob("ad-*.dist-info"))
    if not dist_info_dirs:
        sys.exit("✖ Could not find ad-*.dist-info inside the venv.")
    dist_info = dist_info_dirs[0]

    print(f"➤ Packing wheel from {pkg_dir} (dist-info: {dist_info.name}) …")

    # stage files in a temp dir that looks like an unpacked wheel
    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging)
        copytree(pkg_dir, staging_path / "ad", dirs_exist_ok=True)
        copytree(dist_info, staging_path / dist_info.name, dirs_exist_ok=True)

        # ensure wheel is installed
        sh("python -m pip install --quiet --upgrade 'wheel>=0.38.0'")
        sh(f"python -m wheel pack {staging_path} -d {engine_root}")

    # find and keep original wheel name
    wheel_path = max(engine_root.glob("ad-*.whl"), key=lambda p: p.stat().st_mtime)

    print(f"✓ Created vendored wheel: {wheel_path.name}")


def engine_short_hash() -> str:
    """Return the current commit's 7-char short hash (or die)."""
    try:
        return (subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip())
    except Exception as exc:
        sys.exit(f"✖ Unable to obtain git hash: {exc}")


def engine_branch_name() -> str:
    """Return the current Git branch name of the Engine repo."""
    try:
        return (subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip())
    except Exception as exc:
        sys.exit(f"✖ Unable to obtain git hash: {exc}")


def gpx_short_hash(path: Path) -> str:
    """Return the short Git hash of the GPX repo at the given path."""
    try:
        return (subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=path).decode().strip())
    except Exception as exc:
        sys.exit(f"✖ Unable to obtain GPX git hash at {path}: {exc}")


def gpx_branch_name(path: Path) -> str:
    """Return the current Git branch name of the GPX repo at the given path."""
    try:
        return (subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path).decode().strip())
    except Exception as exc:
        sys.exit(f"✖ Unable to obtain GPX branch name at {path}: {exc}")


@task
def build_docker_image():
    """
    Build & tag the Docker image as test-solve-engine:<engine_branch>-<engine_hash>--<gpx_branch>-<gpx_hash>.
    Expects gpx-zipped.tar.gz and ad-vendored.whl to be present in the Engine directory.
    """
    project_root, engine_root = project_and_engine_roots()
    engine_hash = engine_short_hash()
    engine_branch = engine_branch_name()

    gpx_root = options.gpx.path
    if not gpx_root:
        gpx_root = get_gpx_path(project_root)

    gpx_hash = gpx_short_hash(gpx_root)
    gpx_branch = gpx_branch_name(gpx_root)

    # Sanitize branch names for Docker tag compatibility
    engine_branch_sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '-', engine_branch)
    gpx_branch_sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '-', gpx_branch)

    tag_suffix = f"{engine_branch_sanitized}-{engine_hash}--{gpx_branch_sanitized}-{gpx_hash}"
    local_tag = f"test-solve-engine:{tag_suffix}"

    # Required artefacts
    gpx_tarball = engine_root / "gpx-zipped.tar.gz"
    ad_wheel = engine_root / "ad-1.3.2-py3-none-any.whl"

    if not gpx_tarball.exists():
        sys.exit(f"✖ Required GPX tarball not found: {gpx_tarball}")
    if not ad_wheel.exists():
        sys.exit(f"✖ Required AD wheel not found: {ad_wheel}")

    # Stage vendor/ directory and temporary requirements
    vendor_dir = engine_root / "vendor"
    vendor_dir.mkdir(exist_ok=True)
    staged_wheel = vendor_dir / ad_wheel.name
    copy2(ad_wheel, staged_wheel)

    try:
        engine_rel = engine_root.relative_to(project_root)
        tarball_rel = gpx_tarball.relative_to(project_root)

        env = os.environ.copy()
        env["DOCKER_BUILDKIT"] = "0"

        print(f"➤ Building image {local_tag} …")
        sh(
            "docker build "
            f"--build-arg ENGINE_DIR={engine_rel} "
            f"--build-arg GPX_TARBALL_PATH={tarball_rel} "
            f"-f {engine_root}/Dockerfile.deploy "
            f"-t {local_tag} {project_root}",
            env=env,
        )
        print(f"✓ Image built & tagged: {local_tag}")
    finally:
        staged_wheel.unlink(missing_ok=True)
        rmtree(vendor_dir)
        print("✓ Cleaned up temporary files and restored requirements.txt")


DEFAULT_REGISTRY = "lltest.azurecr.io"


@task
@cmdopts([
    ("login", "l", "Run `az acr login` before pushing"),
])
def push_image_to_azure():
    """
    Push test-solve-engine:<engine_branch>-<engine_hash>--<gpx_branch>-<gpx_hash> to the specified Azure Container Registry.
    """
    engine_hash = engine_short_hash()
    engine_branch = engine_branch_name()

    gpx_root = options.gpx.path
    if not gpx_root:
        gpx_root = get_gpx_path(Path(__file__).resolve().parent.parent)

    gpx_hash = gpx_short_hash(gpx_root)
    gpx_branch = gpx_branch_name(gpx_root)

    # Sanitize branch names for Docker tag compatibility
    engine_branch_sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '-', engine_branch)
    gpx_branch_sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '-', gpx_branch)

    tag_suffix = f"{engine_branch_sanitized}-{engine_hash}--{gpx_branch_sanitized}-{gpx_hash}"
    image_name = "test-solve-engine"
    local_tag = f"{image_name}:{tag_suffix}"

    print("Where should I push the Docker image?")
    print("  [1] lltest.azurecr.io  (default)")
    print("  [2] stage.azurecr.io")
    print("  [3] prod.azurecr.io")

    choice = input("Choose 1 / 2 / 3 [1]: ").strip() or "1"

    if str(choice) == "1":
        registry_url = "lltest.azurecr.io"
    elif str(choice) == "2":
        registry_url = "stage.azurecr.io"
    elif str(choice) == "3":
        registry_url = "prod.azurecr.io"
    else:
        raise SystemExit("✖ Invalid selection—aborting.")

    remote_tag = f"{registry_url}/{image_name}:{tag_suffix}"

    # Optional az acr login
    if getattr(options, "login", False):
        if which("az") is None:
            sys.exit("✖ Azure CLI (az) not found on PATH.")
        print("➤ Logging in to ACR …")
        login_cmd = ["az", "acr", "login", "--name", registry_url.split('.')[0]]
        if subprocess.call(login_cmd) != 0:
            sys.exit("✖ az acr login failed.")

    # Tag image
    print(f"➤ Tagging   {local_tag} → {remote_tag}")
    if subprocess.call(["docker", "tag", local_tag, remote_tag]) != 0:
        sys.exit("✖ docker tag failed — is the image built and tagged correctly?")

    # Push image
    print(f"➤ Pushing   {remote_tag}")
    push = subprocess.run(["docker", "push", remote_tag], text=True, capture_output=True)

    if push.returncode == 0:
        print(f"✓ Image pushed to ACR: {remote_tag}")
        return tag_suffix

    print(push.stdout)
    print(push.stderr)
    sys.exit(
        "✖ docker push failed.\n"
        "   • Make sure you are logged in:  az login --use-device-code\n"
        f"   • Then: az acr login --name {registry_url.split('.')[0]}\n"
        "   • Or ensure your Docker credentials already include this registry."
    )


@task
def deploy_container_revision():
    """
    Deploy a new revision of the container app using a patched YAML file.
    Replaces all <TAG> and <RevisionSuffix> values in the template YAML.
    If a revision with the same suffix exists, appends a timestamp to ensure uniqueness.
    Also prints a formatted deployment summary including the correct revision URL.
    """
    engine_root = Path(__file__).resolve().parent
    repo_root = engine_root.parent
    app_name = 'test-solve-engine'
    resource_group = 'test-solve'
    yaml_path = Path('containerapp.test.yml')
    patched_yaml_path = Path('containerapp.patched.yml')

    # Get paths and metadata
    gpx_root = options.gpx.path or get_gpx_path(repo_root)
    gpx_hash = gpx_short_hash(gpx_root)
    gpx_branch = gpx_branch_name(gpx_root)
    engine_hash = engine_short_hash()
    engine_branch = engine_branch_name()

    # Compose new tag and revision suffix
    tag = f"{engine_branch}-{engine_hash}--{gpx_branch}-{gpx_hash}"
    revision_suffix = engine_hash

    # Function to check if revision already exists
    def revision_exists(suffix):
        revision_name = f"{app_name}--{suffix}"
        result = subprocess.run([
            'az', 'containerapp', 'revision', 'show', '--name', app_name, '--resource-group', resource_group,
            '--revision', revision_name
        ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        return result.returncode == 0

    # Add timestamp if necessary
    if revision_exists(revision_suffix):
        timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        revision_suffix = f"{revision_suffix}-{timestamp}"
        print(f"➤ Existing revision found. Using new revision suffix: {revision_suffix}")
    else:
        print(f"➤ No existing revision found. Using revision suffix: {revision_suffix}")

    # Replace placeholders in YAML
    content = yaml_path.read_text()
    content = content.replace('<TAG>', tag)
    content = content.replace('<RevisionSuffix>', revision_suffix)
    patched_yaml_path.write_text(content)

    try:
        print(f"➤ Deploying revision with tag {tag}...")
        subprocess.run([
            "az", "containerapp", "update", "--name", app_name, "--resource-group", resource_group, "--yaml",
            str(patched_yaml_path)
        ],
                       check=True)
        print("✓ Container app updated with new revision via YAML.")

        # Fetch the deployed revision's FQDN
        fqdn = subprocess.check_output([
            "az", "containerapp", "revision", "show", "--name", app_name, "--resource-group", resource_group,
            "--revision", f"{app_name}--{revision_suffix}", "--query", "properties.fqdn", "--output", "tsv"
        ]).decode().strip()

        # Fetch the Azure subscription ID
        subscription = subprocess.check_output(["az", "account", "show", "--query", "id", "--output",
                                                "tsv"]).decode().strip()

        # Print deployment summary
        print("\n📦 Deployment Summary")
        print("-" * 50)
        print(f"{'Engine Branch':20} : {engine_branch}")
        print(f"{'Engine Hash':20} : {engine_hash}")
        print(f"{'GPX Branch':20} : {gpx_branch}")
        print(f"{'GPX Hash':20} : {gpx_hash}")
        print(f"{'Revision Name':20} : {app_name}--{revision_suffix}")
        print(f"{'Revision URL':20} : https://{fqdn}")
        print(f"{'Subscription ID':20} : {subscription}")
        print("-" * 50)
        print("✓ Deployment successful!")

    finally:
        if patched_yaml_path.exists():
            patched_yaml_path.unlink()
            print("🧹 Removed temporary patched YAML file.")


@task
def deploy():
    build_gpx()
    build_ad()
    build_docker_image()
    push_image_to_azure()
    deploy_container_revision()
    print("✓ Deployment to Azure completed!")


DEFAULT_ACR = "lltest.azurecr.io"


def _stage_vendor_ad(engine_root: Path) -> Path:
    """
    Copy the vendored AD wheel into Engine/vendor/ and return its staged path.
    The caller must clean up (delete vendor dir) afterwards.
    """
    wheel = engine_root / "ad-1.3.2-py3-none-any.whl"
    if not wheel.exists():
        sys.exit("✖ Required AD wheel not found – run `paver build_ad` first.")

    vendor_dir = engine_root / "vendor"
    vendor_dir.mkdir(exist_ok=True)
    staged = vendor_dir / wheel.name
    copy2(wheel, staged)
    return staged


@task
@cmdopts([
    ("registry=", "r", "Target Azure Container Registry (default lltest.azurecr.io)"),
    ("dockerfile=", "f", "Path to Dockerfile (default Engine/Dockerfile.deploy)"),
    ("context=", "c", "Build-context directory (default repo root)"),
])
def az_build_image():
    """
    Build *directly* in Azure ACR via `az acr build`, skipping the local build+push step.
    Produces and pushes the image
        <registry>/test-solve-engine:<engine-branch>-<hash>--<gpx-branch>-<hash>
    """
    project_root, engine_root = project_and_engine_roots()

    # gather hashes / branch names
    engine_hash = engine_short_hash()
    engine_branch = engine_branch_name()

    gpx_root = options.gpx.path or get_gpx_path(project_root)
    gpx_hash = gpx_short_hash(gpx_root)
    gpx_branch = gpx_branch_name(gpx_root)

    # sanitise for Docker tag safety
    sanitise = lambda s: re.sub(r"[^a-zA-Z0-9_.-]", "-", s)
    tag_suffix = (f"{sanitise(engine_branch)}-{engine_hash}"
                  f"--{sanitise(gpx_branch)}-{gpx_hash}")

    image_name = "test-solve-engine"
    registry = getattr(options, "registry", DEFAULT_ACR)
    registry_name = registry.split(".")[0]
    remote_tag = f"{registry}/{image_name}:{tag_suffix}"

    # ensure build artefacts exist
    gpx_tarball = engine_root / "gpx-zipped.tar.gz"
    if not gpx_tarball.exists():
        sys.exit("✖ gpx-zipped.tar.gz not found – run `paver build_gpx` first.")

    staged_wheel = _stage_vendor_ad(engine_root)

    try:
        # build-arg paths relative to repo root, as in local build
        engine_rel = engine_root.relative_to(project_root)
        tarball_rel = gpx_tarball.relative_to(project_root)

        dockerfile_path = Path(getattr(options, "dockerfile", engine_root / "Dockerfile.deploy"))
        context_dir = Path(getattr(options, "context", project_root))

        print(f"➤ Building {remote_tag} in Azure ACR …")
        sh(
            "az acr build "
            f"--registry {registry_name} "
            f"--image {remote_tag} "
            f"--build-arg ENGINE_DIR={engine_rel} "
            f"--build-arg GPX_TARBALL_PATH={tarball_rel} "
            f"--file {dockerfile_path} "
            f"{context_dir}"
        )
        print(f"✓ Image built in ACR as {remote_tag}")

        # Pass tag-suffix on to dependent tasks (optional convenience)
        options.tag_suffix = tag_suffix

    finally:
        # clean up vendor wheel
        rmtree(engine_root / "vendor", ignore_errors=True)
        print("🧹 Cleaned temporary vendor directory")


@task
def az_deploy():
    """
    End-to-end deployment that mirrors `deploy()` but:
      • builds GPX sdist
      • builds AD wheel
      • builds the Docker image *in Azure ACR* (`acr_build_image`)
      • updates the Container App revision
    """
    build_gpx()
    build_ad()
    az_build_image()
    deploy_container_revision()
    print("✓ Azure-build deployment completed!")
