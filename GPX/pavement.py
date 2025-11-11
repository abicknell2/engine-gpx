import os
from typing import Optional

from paver.easy import Bunch, options, sh, task

# Get the directory where the pavement.py file is located
base_dir: str = os.path.abspath(os.path.dirname(__file__))

options(
    mypy=Bunch(output_file="mypy_results.txt"),
    flake8=Bunch(output_file="flake8_results.txt"),
    pylint=Bunch(output_file="pylint_results.txt"),
)


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
    """Format MD files using Prettier"""
    sh("npx prettier --write '**/*.md'")


@task
def pylint() -> None:
    """Run pylint, compare results, and show detailed messages."""
    output_file: str = "pylint_current.txt"
    saved_file: str = options.pylint.output_file

    remove_file(output_file)
    sh(f"pylint $(git ls-files '*.py') --output-format=text > {output_file} || true")

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
    sh(f"pylint $(git ls-files '*.py') --output-format=text > {output_file} || true")
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
    sh(f"mypy {base_dir} > {output_file} || true")

    new_errors, removed_errors = compare_results(output_file, saved_file)

    if new_errors:
        print(f"\n⚠️  {len(new_errors)} new Mypy issues found! 🚨")
        print("".join(new_errors))
    else:
        print("\n✅ No new Mypy issues found! 🎉")

    if removed_errors:
        print(f"\n✅ {len(removed_errors)} Mypy issues have been fixed! 🎉")

    remove_file(output_file)


@task
def mypy_save() -> None:
    """Run mypy and save results."""
    output_file: str = options.mypy.output_file
    remove_file(output_file)
    sh(f"mypy {base_dir} > {output_file} || true")
    print("\n✅ Mypy results have been updated and saved to 'mypy_results.txt'.")


@task
def lint() -> None:
    """Run all linting tasks and print a summary at the end."""
    print("\n🔍 Running Linting Tasks...\n")
    pylint()
    isort()
    prettier()
    flake8()
    mypy()
    yapf()
    print("\n✅ Linting completed!")


@task
def lint_save() -> None:
    """Run all linting tasks and save results."""
    print("\n🔍 Running Linting Tasks and Saving Results...\n")
    pylint_save()
    isort()
    prettier()
    flake8_save()
    mypy_save()
    yapf()
    print("\n✅ Linting completed and results saved!")