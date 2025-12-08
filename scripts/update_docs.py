#!/usr/bin/env python3
"""
VisProbe Documentation Update Tool

Automatically detects code changes and helps update relevant documentation.

Usage:
    python scripts/update_docs.py              # Check current changes
    python scripts/update_docs.py --commit     # Check specific commit
    python scripts/update_docs.py --since main # Check changes since branch
    python scripts/update_docs.py --extract    # Extract docstrings to update API docs
"""

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# Mapping of source files to documentation files
DOC_MAPPING = {
    'src/visprobe/api/decorators.py': [
        'docs/api/index.md',
        'docs/user-guide.md',
        'docs/examples/index.md',
    ],
    'src/visprobe/api/runner.py': [
        'docs/architecture.md',
        'docs/design-rationale.md',
    ],
    'src/visprobe/api/search_modes.py': [
        'docs/user-guide.md',
        'docs/architecture.md',
        'docs/design-rationale.md',
    ],
    'src/visprobe/strategies/base.py': [
        'docs/api/index.md',
        'docs/architecture.md',
    ],
    'src/visprobe/strategies/image.py': [
        'docs/api/index.md',
        'docs/user-guide.md',
        'docs/examples/index.md',
    ],
    'src/visprobe/strategies/adversarial.py': [
        'docs/api/index.md',
        'docs/user-guide.md',
        'docs/examples/index.md',
    ],
    'src/visprobe/properties/base.py': [
        'docs/api/index.md',
    ],
    'src/visprobe/properties/classification.py': [
        'docs/api/index.md',
        'docs/user-guide.md',
        'docs/examples/index.md',
    ],
    'src/visprobe/properties/helpers.py': [
        'docs/api/index.md',
    ],
    'src/visprobe/api/config.py': [
        'docs/api/index.md',
    ],
    'src/visprobe/api/utils.py': [
        'docs/api/index.md',
    ],
}


def get_changed_files(commit: str = None, since: str = None) -> List[str]:
    """Get list of changed Python files using git."""
    try:
        if commit:
            cmd = ['git', 'diff', '--name-only', f'{commit}^', commit]
        elif since:
            cmd = ['git', 'diff', '--name-only', since, 'HEAD']
        else:
            # Get unstaged + staged changes
            cmd = ['git', 'diff', '--name-only', 'HEAD']

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        files = result.stdout.strip().split('\n')

        # Filter for Python files in src/
        return [f for f in files if f.startswith('src/') and f.endswith('.py')]
    except subprocess.CalledProcessError:
        print(f"{Colors.RED}Error: Failed to get git diff. Make sure you're in a git repository.{Colors.END}")
        return []


def map_code_to_docs(changed_files: List[str]) -> Dict[str, Set[str]]:
    """Map changed code files to documentation files that need updating."""
    doc_updates = {}

    for code_file in changed_files:
        # Check if we have a direct mapping
        if code_file in DOC_MAPPING:
            if code_file not in doc_updates:
                doc_updates[code_file] = set()
            doc_updates[code_file].update(DOC_MAPPING[code_file])
        else:
            # Check for pattern matches (e.g., any file in strategies/)
            for pattern, docs in DOC_MAPPING.items():
                if pattern.endswith('*.py'):
                    pattern_dir = pattern.rsplit('/', 1)[0]
                    if code_file.startswith(pattern_dir):
                        if code_file not in doc_updates:
                            doc_updates[code_file] = set()
                        doc_updates[code_file].update(docs)

    return doc_updates


def extract_docstrings(file_path: str) -> Dict[str, str]:
    """Extract docstrings from a Python file."""
    docstrings = {}

    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstrings[node.name] = docstring
    except Exception as e:
        print(f"{Colors.YELLOW}Warning: Could not parse {file_path}: {e}{Colors.END}")

    return docstrings


def extract_function_signatures(file_path: str) -> Dict[str, str]:
    """Extract function signatures from a Python file."""
    signatures = {}

    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Build signature
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    args.append(arg_str)

                return_annotation = ""
                if node.returns:
                    return_annotation = f" -> {ast.unparse(node.returns)}"

                sig = f"def {node.name}({', '.join(args)}){return_annotation}"
                signatures[node.name] = sig
    except Exception as e:
        print(f"{Colors.YELLOW}Warning: Could not parse {file_path}: {e}{Colors.END}")

    return signatures


def print_update_checklist(doc_updates: Dict[str, Set[str]]):
    """Print a checklist of documentation files to update."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}📚 DOCUMENTATION UPDATE CHECKLIST{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

    if not doc_updates:
        print(f"{Colors.GREEN}✅ No documentation updates needed (no Python files changed){Colors.END}\n")
        return

    all_docs = set()
    for docs in doc_updates.values():
        all_docs.update(docs)

    print(f"{Colors.YELLOW}Changed files:{Colors.END}")
    for code_file in sorted(doc_updates.keys()):
        print(f"  • {Colors.CYAN}{code_file}{Colors.END}")

    print(f"\n{Colors.YELLOW}Documentation files to review:{Colors.END}")
    for doc_file in sorted(all_docs):
        exists = "✓" if os.path.exists(doc_file) else "✗"
        color = Colors.GREEN if exists == "✓" else Colors.RED
        print(f"  {color}[{exists}]{Colors.END} {doc_file}")

    print(f"\n{Colors.BOLD}Detailed update guide:{Colors.END}\n")

    for code_file, doc_files in sorted(doc_updates.items()):
        print(f"{Colors.CYAN}📝 {code_file}{Colors.END}")

        # Provide specific guidance based on file type
        if 'decorators.py' in code_file:
            print(f"  {Colors.YELLOW}→{Colors.END} Update decorator signatures in API reference")
            print(f"  {Colors.YELLOW}→{Colors.END} Update usage examples in user guide")
            print(f"  {Colors.YELLOW}→{Colors.END} Verify examples still work")
        elif 'strategies/' in code_file:
            print(f"  {Colors.YELLOW}→{Colors.END} Update strategy class in API reference")
            print(f"  {Colors.YELLOW}→{Colors.END} Add/update usage section in user guide")
            print(f"  {Colors.YELLOW}→{Colors.END} Add complete example if new strategy")
        elif 'properties/' in code_file:
            print(f"  {Colors.YELLOW}→{Colors.END} Update property class in API reference")
            print(f"  {Colors.YELLOW}→{Colors.END} Add/update usage section in user guide")
            print(f"  {Colors.YELLOW}→{Colors.END} Add example showing the property")
        elif 'search_modes.py' in code_file:
            print(f"  {Colors.YELLOW}→{Colors.END} Update search algorithm explanation")
            print(f"  {Colors.YELLOW}→{Colors.END} Update architecture diagrams if needed")
            print(f"  {Colors.YELLOW}→{Colors.END} Update design rationale")
        elif 'runner.py' in code_file:
            print(f"  {Colors.YELLOW}→{Colors.END} Update architecture documentation")
            print(f"  {Colors.YELLOW}→{Colors.END} Update design rationale if logic changed")

        print(f"  {Colors.YELLOW}Files to update:{Colors.END}")
        for doc_file in sorted(doc_files):
            print(f"    • {doc_file}")
        print()

    print(f"{Colors.BOLD}Quick commands:{Colors.END}")
    print(f"  {Colors.CYAN}mkdocs serve{Colors.END}      # Preview documentation")
    print(f"  {Colors.CYAN}mkdocs build --strict{Colors.END}  # Verify docs build")
    print()


def generate_api_doc_template(file_path: str) -> str:
    """Generate API documentation template from code file."""
    docstrings = extract_docstrings(file_path)
    signatures = extract_function_signatures(file_path)

    if not docstrings and not signatures:
        return ""

    template = f"# API Documentation for {file_path}\n\n"

    # Combine signatures and docstrings
    for name in sorted(set(list(docstrings.keys()) + list(signatures.keys()))):
        template += f"## {name}\n\n"

        if name in signatures:
            template += f"```python\n{signatures[name]}\n```\n\n"

        if name in docstrings:
            template += f"{docstrings[name]}\n\n"

        template += "---\n\n"

    return template


def check_docs_build() -> bool:
    """Check if documentation builds successfully."""
    print(f"{Colors.YELLOW}Checking if documentation builds...{Colors.END}")

    try:
        result = subprocess.run(
            ['mkdocs', 'build', '--strict'],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            print(f"{Colors.GREEN}✅ Documentation builds successfully!{Colors.END}")
            return True
        else:
            print(f"{Colors.RED}❌ Documentation build failed:{Colors.END}")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print(f"{Colors.YELLOW}⚠️  mkdocs not found. Install with: pip install mkdocs-material{Colors.END}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='VisProbe Documentation Update Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/update_docs.py              # Check uncommitted changes
  python scripts/update_docs.py --commit HEAD  # Check last commit
  python scripts/update_docs.py --since main # Check changes since main branch
  python scripts/update_docs.py --extract src/visprobe/strategies/image.py
        """
    )

    parser.add_argument('--commit', help='Check specific commit')
    parser.add_argument('--since', help='Check changes since branch/commit')
    parser.add_argument('--extract', help='Extract docstrings from file to generate template')
    parser.add_argument('--check-build', action='store_true', help='Check if docs build successfully')

    args = parser.parse_args()

    # Change to repository root
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    if args.extract:
        # Extract and print template
        template = generate_api_doc_template(args.extract)
        if template:
            print(template)
        else:
            print(f"{Colors.RED}No docstrings found in {args.extract}{Colors.END}")
        return

    if args.check_build:
        success = check_docs_build()
        sys.exit(0 if success else 1)

    # Get changed files
    changed_files = get_changed_files(commit=args.commit, since=args.since)

    if not changed_files:
        print(f"{Colors.GREEN}✅ No Python files changed{Colors.END}")
        return

    # Map to documentation
    doc_updates = map_code_to_docs(changed_files)

    # Print checklist
    print_update_checklist(doc_updates)

    # Optionally check build
    print()
    check_docs_build()


if __name__ == '__main__':
    main()
