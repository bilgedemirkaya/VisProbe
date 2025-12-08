# AI-Powered Documentation Updater

Automatically update documentation using Claude API when code changes.

## Setup (One Time)

### 1. Install Dependencies

```bash
# Install VisProbe with AI documentation support
pip install -e ".[ai-docs]"
```

### 2. Get Claude API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Copy the key (starts with `sk-ant-...`)

### 3. Set API Key

**Option A: Environment Variable (Recommended)**
```bash
# Add to your ~/.bashrc or ~/.zshrc
export ANTHROPIC_API_KEY=sk-ant-api03-...

# Or create .env file (already in .gitignore)
echo "ANTHROPIC_API_KEY=sk-ant-api03-..." > .env
source .env
```

**Option B: One-Time**
```bash
# Set for current session only
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

### 4. Verify Setup

```bash
# Test that it works
visprobe-auto-docs --dry-run
```

## Usage

### Basic Usage

```bash
# Automatically update all docs for changed files
visprobe-auto-docs
```

This will:
1. ✅ Detect changed Python files
2. ✅ Read the code changes
3. ✅ Call Claude API to update documentation
4. ✅ Write updated documentation files

### Interactive Mode (Recommended First Time)

```bash
# Review each change before applying
visprobe-auto-docs --interactive
```

You'll see:
- Diff showing what will change
- Prompt to approve/reject each file
- Option to quit at any time

### Dry Run

```bash
# See what would be updated without making changes
visprobe-auto-docs --dry-run
```

Perfect for:
- Checking what needs updating
- Previewing AI-generated changes
- Testing the tool

### Update Specific File

```bash
# Only update docs for a specific file
visprobe-auto-docs --file strategies/image.py
```

## Workflow Example

### Scenario: You Added a New Strategy

```bash
# 1. Make your code changes
vim src/visprobe/strategies/image.py

# Add your new SaltPepperNoise class...

# 2. Run AI updater in interactive mode
visprobe-auto-docs --interactive

# 3. Review the changes
# AI will show you:
# - Updated API reference
# - New section in user guide
# - Complete code example

# 4. Approve the changes
# Press 'y' for each file

# 5. Verify the updates
mkdocs serve
# Check http://127.0.0.1:8000

# 6. Commit everything
git add src/ docs/
git commit -m "Add SaltPepperNoise strategy with AI-generated docs"
```

## What It Updates

The AI updater knows how to update:

### API Reference (`docs/api/index.md`)
- Function signatures
- Parameter descriptions
- Return types
- Class documentation
- Examples

### User Guide (`docs/user-guide.md`)
- Usage examples
- New feature sections
- Parameter explanations
- Step-by-step guides

### Examples (`docs/examples/index.md`)
- Complete code examples
- Import statements
- Usage patterns
- Real-world scenarios

### Architecture (`docs/architecture.md`)
- Module descriptions
- Design patterns
- Data flow explanations

## How It Works

### 1. Detects Changes
```bash
git diff HEAD
# Finds: src/visprobe/strategies/image.py changed
```

### 2. Maps to Documentation
```
strategies/image.py → docs/api/index.md
                   → docs/user-guide.md
                   → docs/examples/index.md
```

### 3. Calls Claude API
```python
# Sends to Claude:
# - Code diff
# - Current documentation
# - Update instructions
```

### 4. Shows Diff
```diff
+ ## SaltPepperNoise
+
+ ```python
+ class SaltPepperNoise(Strategy):
+     def __init__(self, density=0.05):
+         """Add salt-and-pepper noise to images."""
```

### 5. Applies Changes
Writes updated content to documentation files.

## Cost Estimate

Claude API pricing (as of 2025):
- **Sonnet**: ~$3 per million input tokens, ~$15 per million output tokens

Typical documentation update:
- Input: ~2,000 tokens (code + current docs)
- Output: ~1,000 tokens (updated docs)
- Cost: **~$0.02 per file**

For most updates (3-5 files): **~$0.10 total**

Very affordable! 💰

## Advanced Features

### File Mapping

The tool uses intelligent mapping:

```python
DOC_MAPPING = {
    'src/visprobe/api/decorators.py': [
        'docs/api/index.md',
        'docs/user-guide.md',
        'docs/examples/index.md',
    ],
    # ... more mappings
}
```

### Custom Prompts

The AI receives context-aware prompts:
- API docs get function signature updates
- User guide gets usage examples
- Examples get complete runnable code

### Quality Guarantees

The AI is instructed to:
- ✅ Maintain existing structure
- ✅ Match documentation style
- ✅ Keep examples runnable
- ✅ Preserve accuracy
- ✅ Update only affected sections

## Troubleshooting

### "ANTHROPIC_API_KEY not set"

```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Set it
export ANTHROPIC_API_KEY=sk-ant-...

# Or add to ~/.bashrc
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
source ~/.bashrc
```

### "anthropic package not installed"

```bash
pip install anthropic
# Or
pip install -e ".[ai-docs]"
```

### "No Python files changed"

The tool only detects uncommitted changes. Make sure you have:
```bash
git status  # Shows modified files
```

### API Rate Limits

If you hit rate limits:
```bash
# Wait a few seconds between runs
sleep 5 && visprobe-auto-docs
```

### Bad AI Output

If AI generates incorrect docs:
1. Use `--interactive` to review before applying
2. Reject the change
3. Update manually
4. File an issue with the bad output

## Best Practices

### 1. Always Use Interactive Mode First

```bash
# First time for each change
visprobe-auto-docs --interactive
```

Review AI output before applying!

### 2. Test Documentation Builds

```bash
# After updating
mkdocs build --strict
```

### 3. Preview Before Committing

```bash
mkdocs serve
# Check http://127.0.0.1:8000
```

### 4. Keep Changes Small

Update docs incrementally:
- ✅ One feature at a time
- ✅ Review each update
- ✅ Commit frequently

### 5. Use Dry Run for Planning

```bash
# See what needs updating
visprobe-auto-docs --dry-run
```

## Integration with Git

### Pre-commit Hook

Auto-update docs before each commit:

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash

if git diff --cached --name-only | grep -q '^src/.*\.py$'; then
    echo "Python files changed - updating documentation..."

    # Update docs with AI (interactive mode)
    visprobe-auto-docs --interactive

    # Add updated docs to commit
    git add docs/
fi
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

### GitHub Actions

Auto-update docs in CI:

`.github/workflows/update-docs.yml`:
```yaml
name: Update Documentation

on:
  push:
    branches: [ main ]
    paths:
      - 'src/visprobe/**/*.py'

jobs:
  update-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[ai-docs]"

      - name: Update documentation
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          visprobe-auto-docs

      - name: Create PR
        uses: peter-evans/create-pull-request@v5
        with:
          commit-message: "docs: Auto-update from code changes"
          title: "🤖 Auto-generated documentation update"
          body: "Documentation automatically updated by AI based on code changes"
          branch: auto-docs-update
```

## Comparison with Manual Tool

| Feature | `visprobe-update-docs` | `visprobe-auto-docs` |
|---------|----------------------|---------------------|
| Detects changes | ✅ | ✅ |
| Shows checklist | ✅ | ✅ |
| Extracts docstrings | ✅ | ❌ |
| **Updates files** | ❌ | ✅ |
| **Uses AI** | ❌ | ✅ |
| Requires API key | ❌ | ✅ |
| Interactive mode | ❌ | ✅ |
| Shows diffs | ❌ | ✅ |
| Cost | Free | ~$0.02/file |

**When to use each:**
- **Manual tool**: Just want to see what needs updating
- **AI tool**: Want automatic updates with Claude

## Examples

### Example 1: New Strategy

**Code change:**
```python
class BlurStrategy(Strategy):
    """Apply Gaussian blur to images."""
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size
```

**AI updates API reference:**
```markdown
#### BlurStrategy

```python
class BlurStrategy(Strategy):
    def __init__(self, kernel_size: int = 5)
```

Apply Gaussian blur to images.

**Parameters:**
- `kernel_size`: Size of the blur kernel (default: 5)
```

### Example 2: New Parameter

**Code change:**
```python
@search(
    strategy=...,
    timeout=300,  # NEW!
```

**AI updates user guide:**
```markdown
### Timeout Parameter

Limit search time with the `timeout` parameter:

```python
@search(
    strategy=lambda l: GaussianNoiseStrategy(std=l),
    timeout=300  # Stop after 5 minutes
)
```

## FAQ

**Q: Is my API key secure?**
A: Yes, if you use environment variables. Never commit API keys to git!

**Q: What if AI makes a mistake?**
A: Use `--interactive` mode to review before applying. You can always reject bad updates.

**Q: Can I edit AI-generated docs?**
A: Absolutely! AI provides a starting point. Edit as needed.

**Q: Does it work offline?**
A: No, it requires internet to call Claude API.

**Q: What model does it use?**
A: Claude Sonnet 4 (latest) for best quality and cost balance.

## Tips

1. **Start with `--dry-run`** to see what would change
2. **Use `--interactive`** until you trust the output
3. **Review diffs carefully** before accepting
4. **Keep API key secure** (use environment variables)
5. **Commit docs separately** from code for clear history

---

**Happy AI-powered documenting!** 🤖📚✨
