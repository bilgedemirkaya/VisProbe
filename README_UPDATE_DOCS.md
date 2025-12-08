# Documentation Update Tool

Automatically detect code changes and get guidance on updating documentation.

## Installation

```bash
# Install VisProbe in development mode
pip install -e .
```

This will install the `visprobe-update-docs` command.

## Usage

### Check Current Changes

```bash
# Check uncommitted changes
visprobe-update-docs

# Or run directly
python scripts/update_docs.py
```

### Check Specific Commit

```bash
# Check last commit
visprobe-update-docs --commit HEAD

# Check specific commit
visprobe-update-docs --commit abc123
```

### Check Changes Since Branch

```bash
# Check all changes since main branch
visprobe-update-docs --since main

# Check changes since a tag
visprobe-update-docs --since v0.1.0
```

### Extract Docstrings

Generate API documentation template from a file:

```bash
# Extract docstrings to help write API docs
visprobe-update-docs --extract src/visprobe/strategies/image.py
```

### Check Documentation Build

```bash
# Verify documentation builds correctly
visprobe-update-docs --check-build
```

## Output Example

```
========================================================================
📚 DOCUMENTATION UPDATE CHECKLIST
========================================================================

Changed files:
  • src/visprobe/strategies/image.py

Documentation files to review:
  [✓] docs/api/index.md
  [✓] docs/user-guide.md
  [✓] docs/examples/index.md

Detailed update guide:

📝 src/visprobe/strategies/image.py
  → Update strategy class in API reference
  → Add/update usage section in user guide
  → Add complete example if new strategy
  Files to update:
    • docs/api/index.md
    • docs/examples/index.md
    • docs/user-guide.md

Quick commands:
  mkdocs serve      # Preview documentation
  mkdocs build --strict  # Verify docs build

✅ Documentation builds successfully!
```

## What It Does

The tool automatically:

1. **Detects Changed Files** - Uses git to find modified Python files
2. **Maps to Documentation** - Knows which docs need updating based on code changes
3. **Provides Guidance** - Gives specific instructions for each file type
4. **Checks Build** - Optionally verifies documentation builds correctly
5. **Extracts Docstrings** - Can generate API doc templates from code

## File Mapping

The tool uses these mappings:

| Code File | Documentation Files |
|-----------|---------------------|
| `api/decorators.py` | `api/index.md`, `user-guide.md`, `examples/index.md` |
| `strategies/*.py` | `api/index.md`, `user-guide.md`, `examples/index.md` |
| `properties/*.py` | `api/index.md`, `user-guide.md`, `examples/index.md` |
| `api/runner.py` | `architecture.md`, `design-rationale.md` |
| `api/search_modes.py` | `user-guide.md`, `architecture.md`, `design-rationale.md` |

## Integration with Git Hooks

You can run this automatically before commits:

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash

# Check if any Python files in src/ changed
if git diff --cached --name-only | grep -q '^src/.*\.py$'; then
    echo "Python files changed - checking documentation..."
    python scripts/update_docs.py --check-build

    if [ $? -ne 0 ]; then
        echo ""
        echo "Documentation needs attention!"
        python scripts/update_docs.py
        echo ""
        echo "Please update documentation and run 'mkdocs build --strict'"
        exit 1
    fi
fi
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## CI/CD Integration

### GitHub Actions

Add to `.github/workflows/docs.yml`:

```yaml
- name: Check Documentation
  run: |
    pip install -e .
    visprobe-update-docs --check-build
```

## Workflow Example

### Scenario: You Added a New Strategy

```bash
# 1. Make your code changes
vim src/visprobe/strategies/image.py

# 2. Check what docs need updating
visprobe-update-docs

# Output shows you need to update:
# - docs/api/index.md
# - docs/user-guide.md
# - docs/examples/index.md

# 3. Extract docstrings to help
visprobe-update-docs --extract src/visprobe/strategies/image.py > /tmp/api_template.md

# 4. Update documentation files
vim docs/api/index.md docs/user-guide.md docs/examples/index.md

# 5. Verify docs build
visprobe-update-docs --check-build

# 6. Preview locally
mkdocs serve

# 7. Commit everything
git add src/ docs/
git commit -m "Add SaltPepperNoise strategy with documentation"
```

## Customization

To add new file mappings, edit `scripts/update_docs.py`:

```python
DOC_MAPPING = {
    'src/visprobe/your/new/file.py': [
        'docs/your-doc.md',
    ],
}
```

## Tips

1. **Run After Every Code Change** - Make it a habit
2. **Check Build Before Committing** - Use `--check-build`
3. **Use Extract for New APIs** - Saves time writing API docs
4. **Keep Mappings Updated** - Add new files as you create them

## Troubleshooting

### "git diff failed"
Make sure you're in the VisProbe repository root.

### "mkdocs not found"
Install documentation dependencies:
```bash
pip install -r requirements-docs.txt
```

### No changes detected
The tool only detects Python files in `src/`. Make sure:
- You're checking the right commit/branch
- Files are actually modified in git

## Advanced Usage

### Compare Branches

```bash
# See what docs would need updating in a feature branch
git checkout feature-branch
visprobe-update-docs --since main
```

### Review PR Changes

```bash
# Check what docs a PR modifies
git fetch origin pull/123/head:pr-123
git checkout pr-123
visprobe-update-docs --since main
```

### Generate Documentation Diff

```bash
# See what changed in docs for a commit
git show HEAD:docs/api/index.md > /tmp/before.md
visprobe-update-docs --commit HEAD
# Then manually update docs
diff -u /tmp/before.md docs/api/index.md
```

---

**Happy documenting!** 📚✨
