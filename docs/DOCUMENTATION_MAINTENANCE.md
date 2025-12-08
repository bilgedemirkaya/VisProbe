# Documentation Maintenance Guide

This guide explains how documentation is kept in sync with code changes in VisProbe.

## Automatic Documentation Updates

The `.clinerules` file ensures that Claude automatically updates documentation whenever code changes are made.

## Quick Reference: What Gets Updated When

### When You Add/Modify a Decorator

**Files to Update:**
- ✅ `docs/api/index.md` - Add/update decorator signature and parameters
- ✅ `docs/user-guide.md` - Add/update usage examples
- ✅ `docs/examples/index.md` - Add complete code example

**Example:**
If you add a new parameter to `@search()`:
```python
@search(..., new_param=value)
```

Claude will update:
1. API reference with parameter documentation
2. User guide with usage example
3. At least one complete example showing the parameter

### When You Add/Modify a Strategy

**Files to Update:**
- ✅ `docs/api/index.md` - Strategy class reference
- ✅ `docs/user-guide.md` - Usage section under "Perturbation Strategies"
- ✅ `docs/examples/index.md` - Working example
- ✅ `docs/architecture.md` - If it changes the architecture

**Example:**
Adding `SaltPepperNoise` strategy triggers updates in all three places.

### When You Add/Modify a Property

**Files to Update:**
- ✅ `docs/api/index.md` - Property class reference
- ✅ `docs/user-guide.md` - Usage section under "Robustness Properties"
- ✅ `docs/examples/index.md` - Working example

### When You Change Search Algorithms

**Files to Update:**
- ✅ `docs/architecture.md` - Update flowchart diagrams
- ✅ `docs/user-guide.md` - Update explanation
- ✅ `docs/design-rationale.md` - Update rationale if logic changes

### When You Fix Bugs

**Files to Update:**
- ✅ `docs/user-guide.md` - If documented behavior changes
- ✅ `docs/examples/index.md` - If examples were incorrect

## File Mapping

```
Code Change                          →  Documentation Updates
─────────────────────────────────────────────────────────────────
src/visprobe/api/decorators.py      →  docs/api/index.md
                                        docs/user-guide.md
                                        docs/examples/index.md

src/visprobe/strategies/*.py         →  docs/api/index.md
                                        docs/user-guide.md
                                        docs/examples/index.md

src/visprobe/properties/*.py         →  docs/api/index.md
                                        docs/user-guide.md
                                        docs/examples/index.md

src/visprobe/api/runner.py          →  docs/architecture.md
src/visprobe/api/search_modes.py    →  docs/architecture.md
                                        docs/design-rationale.md
```

## Validation Checklist

Before deploying documentation, verify:

- [ ] `mkdocs build --strict` passes
- [ ] All code examples are runnable
- [ ] No broken links
- [ ] Mermaid diagrams render correctly
- [ ] API signatures match actual code

## Testing Documentation Locally

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Build and check for errors
mkdocs build --strict

# 3. Serve locally
mkdocs serve

# 4. Open http://127.0.0.1:8000/visprobe/

# 5. Check all pages and links
```

## Deploying Updated Documentation

```bash
# 1. Commit documentation changes
git add docs/ mkdocs.yml
git commit -m "Update documentation for [feature/change]"

# 2. Push to GitHub
git push origin main

# 3. Deploy to GitHub Pages
source venv/bin/activate
mkdocs gh-deploy
```

Your documentation will be live at: https://bilgedemirkaya.github.io/visprobe

## What Gets Skipped

Documentation updates are **NOT** required for:
- Internal helper functions (prefixed with `_`)
- Test files in `tests/`
- Build scripts
- CI/CD configuration files
- Minor refactoring with no API changes

## Common Scenarios

### Scenario 1: Adding a New Feature

```
User: "Add support for JPEG compression perturbations"

Claude will:
1. ✅ Implement JPEGCompressionStrategy
2. ✅ Add to docs/api/index.md
3. ✅ Add usage section to docs/user-guide.md
4. ✅ Add example to docs/examples/index.md
5. ✅ Update docs/index.md if it's a major feature
```

### Scenario 2: Fixing a Bug

```
User: "Fix the noise normalization in GaussianNoiseStrategy"

Claude will:
1. ✅ Fix the code
2. ✅ Update docstring
3. ✅ Check if examples need updating
4. ✅ Update user guide if behavior changed
```

### Scenario 3: Changing an API

```
User: "Add 'timeout' parameter to @search decorator"

Claude will:
1. ✅ Add parameter to decorator
2. ✅ Update API reference with full documentation
3. ✅ Add example to user guide
4. ✅ Update at least one example
5. ✅ Note in design rationale why it was added
```

## Troubleshooting

### Documentation Builds but Looks Wrong

Check:
- Mermaid diagrams syntax
- Code fence formatting
- Indentation in nested lists
- Link paths (relative vs absolute)

### Links Are Broken

Check:
- File paths are correct
- Files exist in `docs/` directory
- Links use `.md` extension
- Navigation in `mkdocs.yml` matches files

### Examples Don't Work

Check:
- Imports are included
- Code is complete (not snippets)
- Variable names are defined
- Examples match current API

## Best Practices

1. **Update docstrings first** - They inform the documentation
2. **Test examples** - Make sure code actually runs
3. **Use consistent style** - Follow existing documentation patterns
4. **Keep it simple** - Don't over-complicate explanations
5. **Show, don't tell** - Examples > explanations

## Quick Commands

```bash
# Activate environment
source venv/bin/activate

# Build documentation
mkdocs build --strict

# Serve locally
mkdocs serve

# Deploy to GitHub Pages
mkdocs gh-deploy

# Check what would be deployed
mkdocs build
cd site && python -m http.server 8000
```

## Version Control

Good commit messages for documentation:
```bash
# Good
git commit -m "docs: Add JPEGCompression strategy to user guide and examples"
git commit -m "docs: Update search algorithm flowchart in architecture"
git commit -m "docs: Fix broken links in API reference"

# Not ideal
git commit -m "Update docs"
git commit -m "Documentation changes"
```

## Getting Help

If documentation seems out of sync:
1. Check `.clinerules` for the policy
2. Review this maintenance guide
3. Look at recent documentation commits for patterns
4. Ask: "Can you review and update the documentation for [feature]?"

---

**Remember: Documentation is part of the feature, not an afterthought!**
