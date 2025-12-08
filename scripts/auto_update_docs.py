#!/usr/bin/env python3
"""
VisProbe AI-Powered Documentation Updater

Automatically updates documentation using Claude API when code changes.

Usage:
    # Set API key once
    export ANTHROPIC_API_KEY=your_key_here

    # Run automatic update
    visprobe-auto-docs

    # With interactive confirmation
    visprobe-auto-docs --interactive

    # Dry run (show what would be updated)
    visprobe-auto-docs --dry-run
"""

import argparse
import difflib
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    from anthropic import Anthropic
except ImportError:
    print("Error: anthropic package not installed.")
    print("Install with: pip install anthropic")
    sys.exit(1)


class Colors:
    """ANSI color codes."""
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
    ],
    'src/visprobe/api/search_modes.py': [
        'docs/user-guide.md',
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
    'src/visprobe/properties/classification.py': [
        'docs/api/index.md',
        'docs/user-guide.md',
        'docs/examples/index.md',
    ],
}


def get_changed_files() -> List[str]:
    """Get list of changed Python files using git."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        files = result.stdout.strip().split('\n')
        return [f for f in files if f.startswith('src/') and f.endswith('.py')]
    except subprocess.CalledProcessError:
        return []


def get_file_diff(file_path: str) -> str:
    """Get git diff for a specific file."""
    try:
        result = subprocess.run(
            ['git', 'diff', 'HEAD', file_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return ""


def read_file(file_path: str) -> str:
    """Read file contents."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def create_update_prompt(code_file: str, doc_file: str, code_diff: str, current_doc: str) -> str:
    """Create prompt for Claude to update documentation."""

    # Determine the type of documentation file
    doc_type = "API reference"
    if "user-guide" in doc_file:
        doc_type = "user guide"
    elif "examples" in doc_file:
        doc_type = "examples"
    elif "architecture" in doc_file:
        doc_type = "architecture documentation"

    prompt = f"""You are updating the {doc_type} for VisProbe based on code changes.

CODE FILE: {code_file}

CODE CHANGES (git diff):
```diff
{code_diff}
```

CURRENT DOCUMENTATION ({doc_file}):
```markdown
{current_doc}
```

TASK:
Update the documentation to reflect the code changes. Follow these rules:

1. **Maintain existing structure** - Don't reorganize unless necessary
2. **Update changed sections** - Only modify parts affected by code changes
3. **Keep examples working** - If you update examples, make sure they're complete and runnable
4. **Preserve style** - Match the existing documentation style
5. **Be accurate** - Documentation must match the actual code

SPECIFIC GUIDELINES:

For API Reference (docs/api/index.md):
- Update function signatures if they changed
- Update parameter descriptions
- Add new functions/classes
- Remove deprecated ones

For User Guide (docs/user-guide.md):
- Update usage examples
- Add sections for new features
- Update parameters in examples
- Keep explanations clear and concise

For Examples (docs/examples/index.md):
- Ensure all code examples work
- Add examples for new features
- Update imports if needed
- Make examples complete and runnable

For Architecture (docs/architecture.md):
- Update diagrams if structure changed
- Update module descriptions
- Keep diagrams simple and clear

OUTPUT:
Provide ONLY the updated documentation content. Do not include explanations or comments outside the documentation itself.
Start your response with the updated markdown content.
"""
    return prompt


def update_doc_with_claude(
    code_file: str,
    doc_file: str,
    code_diff: str,
    client: Anthropic
) -> Optional[str]:
    """Use Claude to update documentation."""

    print(f"{Colors.CYAN}Processing {doc_file}...{Colors.END}")

    # Read current documentation
    current_doc = read_file(doc_file)
    if not current_doc:
        print(f"{Colors.YELLOW}  Warning: {doc_file} not found, skipping{Colors.END}")
        return None

    # Create prompt
    prompt = create_update_prompt(code_file, doc_file, code_diff, current_doc)

    try:
        # Call Claude API
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        updated_content = message.content[0].text
        return updated_content

    except Exception as e:
        print(f"{Colors.RED}  Error calling Claude API: {e}{Colors.END}")
        return None


def show_diff(original: str, updated: str, file_path: str):
    """Show colored diff between original and updated content."""
    original_lines = original.splitlines(keepends=True)
    updated_lines = updated.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        updated_lines,
        fromfile=f"{file_path} (original)",
        tofile=f"{file_path} (updated)",
        lineterm=''
    )

    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}Changes to {file_path}:{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            print(f"{Colors.GREEN}{line}{Colors.END}")
        elif line.startswith('-') and not line.startswith('---'):
            print(f"{Colors.RED}{line}{Colors.END}")
        elif line.startswith('@'):
            print(f"{Colors.CYAN}{line}{Colors.END}")
        else:
            print(line)


def write_file(file_path: str, content: str):
    """Write content to file."""
    with open(file_path, 'w') as f:
        f.write(content)


def confirm_update(file_path: str) -> bool:
    """Ask user to confirm update."""
    while True:
        response = input(f"\n{Colors.YELLOW}Apply changes to {file_path}? (y/n/q): {Colors.END}").lower()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        elif response == 'q':
            print(f"{Colors.RED}Aborted by user{Colors.END}")
            sys.exit(0)
        else:
            print("Please enter 'y' (yes), 'n' (no), or 'q' (quit)")


def main():
    parser = argparse.ArgumentParser(
        description='AI-Powered Documentation Updater for VisProbe',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  visprobe-auto-docs                    # Auto-update with prompts
  visprobe-auto-docs --interactive      # Review each change
  visprobe-auto-docs --dry-run          # See what would be updated
  visprobe-auto-docs --file strategies/image.py  # Update docs for specific file

Environment Variables:
  ANTHROPIC_API_KEY    Your Claude API key (required)
        """
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Show diffs and ask for confirmation before each update'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    parser.add_argument(
        '--file',
        help='Update docs for a specific file only'
    )

    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print(f"{Colors.RED}Error: ANTHROPIC_API_KEY environment variable not set{Colors.END}")
        print("\nSet your API key with:")
        print(f"  {Colors.CYAN}export ANTHROPIC_API_KEY=your_key_here{Colors.END}")
        print("\nGet your API key at: https://console.anthropic.com/")
        sys.exit(1)

    # Initialize Claude client
    client = Anthropic(api_key=api_key)

    # Change to repository root
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}🤖 AI-Powered Documentation Updater{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

    # Get changed files
    if args.file:
        changed_files = [f"src/{args.file}" if not args.file.startswith('src/') else args.file]
    else:
        changed_files = get_changed_files()

    if not changed_files:
        print(f"{Colors.GREEN}✅ No Python files changed{Colors.END}\n")
        return

    print(f"{Colors.YELLOW}Changed files:{Colors.END}")
    for f in changed_files:
        print(f"  • {f}")
    print()

    # Process each changed file
    updates_made = 0

    for code_file in changed_files:
        # Get git diff
        code_diff = get_file_diff(code_file)
        if not code_diff:
            print(f"{Colors.YELLOW}No diff available for {code_file}, skipping{Colors.END}")
            continue

        # Find documentation files to update
        doc_files = DOC_MAPPING.get(code_file, [])
        if not doc_files:
            print(f"{Colors.YELLOW}No documentation mapping for {code_file}{Colors.END}")
            continue

        print(f"\n{Colors.BOLD}Processing {code_file}:{Colors.END}")

        for doc_file in doc_files:
            # Generate updated documentation
            updated_content = update_doc_with_claude(code_file, doc_file, code_diff, client)

            if not updated_content:
                continue

            # Read original
            original_content = read_file(doc_file)

            # Show diff if interactive or dry-run
            if args.interactive or args.dry_run:
                show_diff(original_content, updated_content, doc_file)

            # Apply changes
            if args.dry_run:
                print(f"{Colors.YELLOW}  [DRY RUN] Would update {doc_file}{Colors.END}")
            elif args.interactive:
                if confirm_update(doc_file):
                    write_file(doc_file, updated_content)
                    print(f"{Colors.GREEN}  ✅ Updated {doc_file}{Colors.END}")
                    updates_made += 1
                else:
                    print(f"{Colors.YELLOW}  ⏭️  Skipped {doc_file}{Colors.END}")
            else:
                write_file(doc_file, updated_content)
                print(f"{Colors.GREEN}  ✅ Updated {doc_file}{Colors.END}")
                updates_made += 1

    # Summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    if args.dry_run:
        print(f"{Colors.YELLOW}Dry run complete - no files were modified{Colors.END}")
    else:
        print(f"{Colors.GREEN}✅ Updated {updates_made} documentation file(s){Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

    if updates_made > 0 and not args.dry_run:
        print(f"{Colors.YELLOW}Next steps:{Colors.END}")
        print(f"  1. Review the changes: {Colors.CYAN}git diff docs/{Colors.END}")
        print(f"  2. Preview docs: {Colors.CYAN}mkdocs serve{Colors.END}")
        print(f"  3. Commit changes: {Colors.CYAN}git add docs/ && git commit{Colors.END}")
        print()


if __name__ == '__main__':
    main()
