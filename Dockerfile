# windbreak container image (issue #15). Minimal, non-root, runs the CLI.
FROM python:3.12-slim

# Do not buffer stdout/stderr so the JSON log stream is emitted promptly.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install the project (and its console script) from source. The build context
# is the repo root (deploy/docker-compose.yml, issue #445), so /app holds the
# repository -- which is what makes the services' repo-relative fixture paths
# (`tests/fixtures/...`) resolve. See .dockerignore for what is left out.
COPY . .
RUN pip install --no-cache-dir .

# Drop root: create an unprivileged user and switch to it after install so a
# compromised process cannot escalate. SPEC-aligned defense in depth.
RUN useradd --create-home --uid 10001 windbreak

# Pre-create the volume mount points and give them to that user. Docker seeds a
# fresh named volume from the image's directory at the mount path; if the image
# has no such directory it creates one owned by root, and the non-root process
# above cannot write it -- a second, quieter reason the ledger volume would
# stay empty (issue #446).
RUN mkdir -p /var/lib/windbreak/ledger /var/lib/windbreak/reports \
    && chown -R windbreak:windbreak /var/lib/windbreak

USER windbreak

# Default to the pipeline process; compose/systemd override --process per unit.
CMD ["windbreak", "run"]
