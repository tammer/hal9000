#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from ftplib import FTP, FTP_TLS, error_perm
from pathlib import Path

from dotenv import load_dotenv

from generate_website import resolve_website_dir


@dataclass
class DeployConfig:
    host: str
    port: int
    username: str
    password: str
    use_tls: bool


def resolve_deploy_config() -> DeployConfig:
    host = os.getenv("DEPLOY_HOST")
    if not host:
        raise ValueError(
            "DEPLOY_HOST is not set. "
            "Set it to your ICDSoft server hostname (see your Welcome e-mail or Control Panel)."
        )

    username = os.getenv("DEPLOY_USER")
    if not username:
        raise ValueError(
            "DEPLOY_USER is not set. Set it to your FTP username (e.g. antler)."
        )

    password = os.getenv("DEPLOY_PASSWORD")
    if not password:
        raise ValueError(
            "DEPLOY_PASSWORD is not set. Set it to the password for your FTP user."
        )

    use_tls = os.getenv("DEPLOY_USE_TLS", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }

    return DeployConfig(
        host=host,
        port=int(os.getenv("DEPLOY_PORT", "21")),
        username=username,
        password=password,
        use_tls=use_tls,
    )


def connect(config: DeployConfig) -> FTP:
    if config.use_tls:
        ftp: FTP = FTP_TLS()
    else:
        ftp = FTP()

    ftp.connect(config.host, config.port)
    ftp.login(config.username, config.password)
    if isinstance(ftp, FTP_TLS):
        ftp.prot_p()
    return ftp


def remote_html_files(ftp: FTP) -> set[str]:
    try:
        names = ftp.nlst()
    except error_perm as exc:
        # 550 with no entries just means an empty directory.
        if str(exc).startswith("550"):
            return set()
        raise
    return {name for name in names if name.endswith(".html")}


def deploy_website(website_dir: Path | None = None) -> int:
    load_dotenv()

    try:
        config = resolve_deploy_config()
        if website_dir is None:
            website_dir = resolve_website_dir()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not website_dir.is_dir():
        print(
            f"Error: website directory does not exist: {website_dir}. "
            "Run generate_website.py first.",
            file=sys.stderr,
        )
        return 1

    local_files = sorted(p for p in website_dir.glob("*.html") if p.is_file())
    if not local_files:
        print(f"Error: no HTML files found in {website_dir}", file=sys.stderr)
        return 1

    try:
        ftp = connect(config)
    except (OSError, error_perm) as exc:
        print(f"Error: could not connect to {config.host}: {exc}", file=sys.stderr)
        return 1

    try:
        # The FTP user lands directly in the target directory, so we upload
        # into the current working directory without changing paths.
        uploaded = 0
        local_names: set[str] = set()
        for path in local_files:
            with path.open("rb") as handle:
                ftp.storbinary(f"STOR {path.name}", handle)
            local_names.add(path.name)
            uploaded += 1
            print(f"Uploaded {path.name}", file=sys.stderr)

        removed = 0
        for name in remote_html_files(ftp) - local_names:
            ftp.delete(name)
            removed += 1
            print(f"Removed stale {name}", file=sys.stderr)
    except (OSError, error_perm) as exc:
        print(f"Error: deployment failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            ftp.quit()
        except (OSError, error_perm):
            ftp.close()

    protocol = "FTPS" if config.use_tls else "FTP"
    print(
        f"Done: uploaded {uploaded} file(s) and removed {removed} stale file(s) "
        f"via {protocol} to {config.username}@{config.host}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(deploy_website())
