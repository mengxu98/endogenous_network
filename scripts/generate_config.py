#!/usr/bin/env python3
import os
import ast
import sys
from pathlib import Path
from typing import Set, Dict


class DependencyFinder(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()

    def visit_Import(self, node):
        for name in node.names:
            self.imports.add(name.name.split(".")[0])

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module.split(".")[0])


def find_python_files(directory: str) -> Set[Path]:
    """Recursively find all Python files"""
    python_files = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.add(Path(root) / file)
    return python_files


def get_imports_from_file(file_path: Path) -> Set[str]:
    """Extract imported packages from Python file"""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
            finder = DependencyFinder()
            finder.visit(tree)
            return finder.imports
        except:
            print(f"Warning: Could not parse {file_path}")
            return set()


def filter_local_and_stdlib_modules(
    imports: Set[str], local_modules: Set[str]
) -> Set[str]:
    """Filter out standard library modules and local modules"""
    stdlib_modules = set(sys.stdlib_module_names)
    return {
        imp
        for imp in imports
        if imp not in stdlib_modules
        and imp not in local_modules
        and not imp.endswith(".py")
    }


def find_local_modules(directory: Path) -> Set[str]:
    """Find all local Python module names"""
    local_modules = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                module_name = Path(file).stem
                local_modules.add(module_name)
    return local_modules


def generate_requirements_txt(imports: Set[str]) -> str:
    """Generate requirements.txt content"""
    # Predefined package versions
    common_versions = {
        "numpy": ">=1.21.0",
        "pandas": ">=1.3.0",
        "scipy": ">=1.7.0",
        "matplotlib": ">=3.4.0",
        "seaborn": ">=0.11.0",
        "networkx": ">=2.6.0",
        "sklearn": "",  # scikit-learn import name is sklearn
        "tqdm": "",
        "graphviz": "",
    }

    lines = []
    for package in sorted(imports):
        version = common_versions.get(package, "")
        lines.append(f"{package}{version}")

    return "\n".join(lines)


def main():
    # Get project root directory
    project_root = Path(__file__).parent.parent

    # Find all local modules
    local_modules = find_local_modules(project_root)

    # Find all Python files
    python_files = find_python_files(project_root)

    # Collect all imports
    all_imports = set()
    for file in python_files:
        imports = get_imports_from_file(file)
        all_imports.update(imports)

    # Filter standard library and local modules
    external_imports = filter_local_and_stdlib_modules(all_imports, local_modules)

    # Generate requirements.txt
    requirements_content = generate_requirements_txt(external_imports)

    # Write to file
    with open(project_root / "requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content)
    print("Generated requirements.txt")


if __name__ == "__main__":
    main()
