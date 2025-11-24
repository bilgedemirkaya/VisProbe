# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of VisProbe seriously. If you discover a security vulnerability, please follow these steps:

### Where to Report

**DO NOT** open a public issue for security vulnerabilities.

Instead, please report security vulnerabilities via:
- Email: bilgedemirkaya@example.com
- GitHub Security Advisories: https://github.com/bilgedemirkaya/VisProbe/security/advisories/new

For critical vulnerabilities, please use GitHub Security Advisories for secure, private reporting.

### What to Include

Please include the following information in your report:
- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if available)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: We will acknowledge your report within 48 hours
- **Status Update**: We will provide a detailed response within 7 days indicating next steps
- **Fix Timeline**: We aim to release a patch within 30 days for critical vulnerabilities

### Disclosure Policy

- We request that you do not publicly disclose the vulnerability until we have released a fix
- We will credit you in our release notes (unless you prefer to remain anonymous)
- Once the vulnerability is patched, we will publish a security advisory

## Security Best Practices

When using VisProbe:

1. **Keep dependencies updated**: Regularly update PyTorch, NumPy, and other dependencies
2. **Use virtual environments**: Isolate VisProbe installations to prevent dependency conflicts
3. **Validate inputs**: When testing custom models, ensure they come from trusted sources
4. **Monitor resource usage**: Adversarial testing can be computationally intensive
5. **Secure results**: Test results may contain sensitive information about model vulnerabilities

## Known Security Considerations

- **Model Loading**: VisProbe loads PyTorch models. Only use models from trusted sources.
- **Data Processing**: Image data is processed using PIL/Pillow. Keep these libraries updated.
- **Subprocess Execution**: The CLI spawns Python subprocesses. Ensure Python interpreter path is secure.

Thank you for helping keep VisProbe and its users safe!
