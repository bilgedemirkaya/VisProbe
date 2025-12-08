# Deploying Documentation to GitHub Pages

This guide shows you how to deploy your VisProbe documentation to GitHub Pages.

## Prerequisites

1. Python 3.7 or higher
2. GitHub account
3. Git installed

## Method 1: Using MkDocs (Recommended)

### Step 1: Install MkDocs and Dependencies

```bash
cd /path/to/VisProbe

# Install MkDocs with Material theme
pip install mkdocs mkdocs-material pymdown-extensions mkdocs-mermaid2-plugin
```

### Step 2: Test Locally

```bash
# Serve documentation locally
mkdocs serve

# Open browser to http://127.0.0.1:8000
```

### Step 3: Build and Deploy to GitHub Pages

```bash
# Build the documentation
mkdocs build

# This creates a 'site/' directory with static HTML

# Deploy to GitHub Pages (gh-pages branch)
mkdocs gh-deploy
```

This command will:
1. Build your documentation
2. Create/update the `gh-pages` branch
3. Push it to GitHub
4. Your site will be live at `https://yourusername.github.io/visprobe`

### Step 4: Configure GitHub Repository Settings

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under "Source", select:
   - Branch: `gh-pages`
   - Folder: `/ (root)`
4. Click **Save**

Your documentation will be live in a few minutes!

## Method 2: Using GitHub Actions (Automated)

### Step 1: Create GitHub Actions Workflow

Create `.github/workflows/docs.yml`:

```yaml
name: Deploy Documentation

on:
  push:
    branches:
      - main
  pull_request:

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure Git Credentials
        run: |
          git config user.name github-actions[bot]
          git config user.email 41898282+github-actions[bot]@users.noreply.github.com

      - uses: actions/setup-python@v4
        with:
          python-version: 3.x

      - name: Install dependencies
        run: |
          pip install mkdocs-material
          pip install pymdown-extensions
          pip install mkdocs-mermaid2-plugin

      - name: Deploy documentation
        run: mkdocs gh-deploy --force
```

### Step 2: Commit and Push

```bash
git add .github/workflows/docs.yml
git add mkdocs.yml
git add docs/
git commit -m "Add documentation and deployment workflow"
git push origin main
```

Your documentation will automatically rebuild and deploy on every push to `main`!

## Method 3: Using Custom Domain

### Step 1: Add CNAME File

Create `docs/CNAME`:

```
docs.yourproject.com
```

### Step 2: Configure DNS

Add a CNAME record in your DNS provider:

```
docs.yourproject.com CNAME yourusername.github.io
```

### Step 3: Update mkdocs.yml

```yaml
site_url: https://docs.yourproject.com
```

### Step 4: Deploy

```bash
mkdocs gh-deploy
```

Wait for DNS propagation (5-60 minutes).

## Customization

### Updating the Theme

Edit `mkdocs.yml`:

```yaml
theme:
  name: material
  palette:
    primary: indigo  # Change to: blue, red, green, etc.
    accent: indigo
  font:
    text: Roboto
    code: Roboto Mono
```

### Adding Custom CSS

1. Create `docs/stylesheets/extra.css`:

```css
:root {
    --md-primary-fg-color: #4a9eff;
}

.md-header {
    background-color: linear-gradient(to right, #4a9eff, #66bb6a);
}
```

2. Reference in `mkdocs.yml`:

```yaml
extra_css:
  - stylesheets/extra.css
```

### Adding Google Analytics

Edit `mkdocs.yml`:

```yaml
extra:
  analytics:
    provider: google
    property: G-XXXXXXXXXX  # Your tracking ID
```

## Troubleshooting

### Build Fails with "No module named 'mermaid2'"

```bash
pip install mkdocs-mermaid2-plugin
```

### 404 Error on GitHub Pages

1. Check that `gh-pages` branch exists
2. Verify GitHub Pages is enabled in repository settings
3. Wait a few minutes for deployment to complete

### Images Not Loading

Ensure image paths are relative:

```markdown
<!-- Good -->
![Architecture](diagrams/architecture.png)

<!-- Bad -->
![Architecture](/diagrams/architecture.png)
```

### Mermaid Diagrams Not Rendering

1. Install plugin: `pip install mkdocs-mermaid2-plugin`
2. Add to `mkdocs.yml`:

```yaml
plugins:
  - mermaid2
```

## Alternative: ReadTheDocs

If you prefer ReadTheDocs:

### Step 1: Create `requirements.txt`

```txt
mkdocs>=1.5.0
mkdocs-material>=9.0.0
pymdown-extensions>=10.0.0
mkdocs-mermaid2-plugin>=0.6.0
```

### Step 2: Create `.readthedocs.yml`

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

mkdocs:
  configuration: mkdocs.yml

python:
  install:
    - requirements: requirements.txt
```

### Step 3: Connect to ReadTheDocs

1. Go to [readthedocs.org](https://readthedocs.org)
2. Sign in with GitHub
3. Import your repository
4. Documentation will build automatically

Your docs will be at: `https://visprobe.readthedocs.io`

## Continuous Integration

### Pre-commit Hook for Local Testing

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
mkdocs build --strict || exit 1
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

Now documentation will be validated before every commit!

## Summary

**Quick Start:**
```bash
# Install
pip install mkdocs-material pymdown-extensions mkdocs-mermaid2-plugin

# Test locally
mkdocs serve

# Deploy to GitHub Pages
mkdocs gh-deploy
```

**Your documentation is now live!**
- URL: `https://yourusername.github.io/visprobe`
- Updates automatically with GitHub Actions
- Professional appearance with Material theme
- Searchable, responsive, and fast

## Next Steps

1. Update `mkdocs.yml` with your GitHub username
2. Add more examples to `docs/examples/`
3. Include API diagrams in `docs/diagrams/`
4. Enable Google Analytics for visitor tracking
5. Add versioning for different releases

Happy documenting! 📚
