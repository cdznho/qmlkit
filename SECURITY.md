# Security Policy

qmlkit is a public-alpha ML SDK. It does not execute remote code, manage
credentials, or connect to quantum hardware by default.

## Supported Versions

Security fixes target the latest public version on `main`.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately by opening a GitHub security
advisory on the repository, or by contacting the maintainer through GitHub.

Do not disclose security issues publicly until there has been time to assess
and patch them.

## Scope

In scope:

- dependency-related vulnerabilities;
- unsafe file or network behavior introduced by qmlkit;
- examples or docs that encourage insecure usage.

Out of scope:

- vulnerabilities in optional third-party frameworks unless qmlkit usage makes
  them materially worse;
- benchmark-quality issues or model-performance concerns.
