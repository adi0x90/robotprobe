# Publishing RobotProbe

The PyPI project name is `robotprobe`. Releases are built from `main` in
[`adi0x90/robotprobe`](https://github.com/adi0x90/robotprobe) by the manually run
[`publish.yml`](../.github/workflows/publish.yml) workflow.

## First-time account setup

Create a PyPI account, verify its email address, and enable two-factor
authentication. In [PyPI account publishing settings](https://pypi.org/manage/account/publishing/),
add a pending GitHub publisher with these exact values:

| Field | Value |
| --- | --- |
| PyPI project name | `robotprobe` |
| Owner | `adi0x90` |
| Repository name | `robotprobe` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

The workflow uses short-lived GitHub OIDC credentials; no PyPI API token is needed.
The first successful upload creates the project under the PyPI account that
registered the pending publisher. Registering a pending publisher alone does not
reserve the package name. See [PyPI's setup guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Release

1. Set a new version in `pyproject.toml`, refresh `uv.lock`, and merge the release
   content to `main`. Keep the alpha version suffix while the project is in alpha.
2. Check that the repository's test workflow passed for that commit.
3. Open **Actions → Publish to PyPI → Run workflow**, select `main`, and leave
   **Publish the checked distributions to PyPI** unchecked for a build-only rehearsal.
4. Run the workflow again with that checkbox checked to publish. It builds a wheel
   from the source archive, checks package metadata, and tests the installed wheel
   outside the checkout before the publishing job can run.
5. Verify the version on [PyPI](https://pypi.org/project/robotprobe/) and install it
   in a fresh environment:

   ```bash
   python -m pip install 'robotprobe==0.2.0a0'
   robotprobe demo
   ```

Use the actual released version in the installation command. Uploaded filenames
cannot be reused; corrections to a published package require a new version.

To install the latest alpha release, use `python -m pip install --pre robotprobe`.
