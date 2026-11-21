"""Build the Windows x64 release ZIP with a local CPython runtime and core dependencies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parent.parent
RELEASE_DIR = ROOT / ".runtime" / "release"
LOCK_FILE = ROOT / "scripts" / "requirements-portable-win-x64.txt"
ZIP_NAME = "AI-Interview-Demo-windows-x64.zip"
PYTHON_ARCHIVE = "cpython-3.11.18+20261110-x86_64-pc-windows-msvc-pgo-full.tar.zst"
PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20261110/"
    + PYTHON_ARCHIVE.replace("+", "%2B")
)
PYTHON_SHA256 = "0dd076313373790b1d068b15c005964c58c95b6ca2c6602190ea29802f457f49"
STREAMLIT_LICENSE_URL = "https://raw.githubusercontent.com/streamlit/streamlit/1.67.0/LICENSE"
STREAMLIT_LICENSE_SHA256 = "c95bae1d1ce0235ecccd3560b772ec1efb97f348a79f0fbe0a634f0c2ccefe2c"

PROJECT_FILES = (
    "Run-Demo.cmd", "README.md", ".streamlit/config.toml", "app.py", "settings.py", "llm.py",
    "business_assistant.py", "chat_history.py", "classifier.py", "common.py", "quote_agent.py", "rag_engine.py",
    "requirements.txt", "requirements-models.txt", "train_macbert.py", "start-demo.ps1",
    "pages_ui/__init__.py", "pages_ui/classification_page.py", "pages_ui/quote_page.py", "pages_ui/rag_page.py",
    "pages_ui/chat.css", "pages_ui/workspace.css", "pages_ui/assets/toner-workspace.webp",
    "data/categories.json", "data/faq.json", "data/products_demo.xlsx", "data/products_test.csv",
    "data/products_train.csv", "data/quotation.xlsx", "data/quotation.csv", "data/sku.csv",
    "data/after_sales/售后政策.txt", "data/after_sales/爱普生 L3158 排障手册.txt",
    "data/after_sales/佳能 LBP2900 排障手册.txt", "data/after_sales/惠普 M404 维修手册.txt",
    "data/selection/佳能和爱普生耗材兼容表.txt", "data/selection/惠普耗材兼容表.txt",
)


def _download(url: str, destination: Path) -> Path:
    with urlopen(url, timeout=90) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target, length=1024 * 1024)
    return destination


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _copy_project(destination: Path) -> None:
    for relative in PROJECT_FILES:
        source = (ROOT / relative).resolve(strict=True)
        if not source.is_file() or not source.is_relative_to(ROOT):
            raise ValueError(f"Portable source path is invalid: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _copy_runtime(source: Path, runtime: Path) -> None:
    installation = source / "install"
    runtime.mkdir()
    for binary in sorted(installation.glob("*.dll")):
        shutil.copyfile(binary, runtime / binary.name)
    for name in ("python.exe", "pythonw.exe", "LICENSE.txt"):
        shutil.copyfile(installation / name, runtime / name)
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pdb", "site-packages", "ensurepip", "test")
    for directory in ("DLLs", "Lib", "tcl"):
        shutil.copytree(installation / directory, runtime / directory, ignore=ignored)
    shutil.copytree(source / "licenses", runtime / "licenses")
    shutil.copyfile(source / "PYTHON.json", runtime / "PYTHON-BUILD.json")


def _install_dependencies(runtime: Path, offline: bool) -> Path:
    site = runtime / "Lib" / "site-packages"
    command = ["uv", "pip", "sync", "--no-config", "--no-build", "--link-mode", "copy"]
    if offline:
        command.append("--offline")
    command.extend(["--python", str(runtime / "python.exe"), "--target", str(site), str(LOCK_FILE)])
    subprocess.run(command, check=True)
    return site


def _write_notices(destination: Path, site: Path, streamlit_license: Path) -> None:
    licenses = destination / "runtime" / "licenses"
    shutil.copyfile(streamlit_license, licenses / "LICENSE.streamlit.txt")
    manifest = [
        "AI-Interview-Demo portable Windows x64 third-party notices", "",
        "Python 3.11.18 from Astral Python Build Standalone (2026-11-10)",
        f"Distribution: {PYTHON_URL}", f"SHA-256: {PYTHON_SHA256}",
        "Python and runtime library licenses: runtime/LICENSE.txt and runtime/licenses/.", "",
        "Core Python dependencies (licenses and notices as distributed with each wheel):", "",
    ]
    packages = sorted(importlib.metadata.distributions(path=[str(site)]), key=lambda d: d.metadata["Name"].lower())
    for package in packages:
        name = package.metadata["Name"]
        paths = [
            str(file) for file in (package.files or ())
            if "licenses" in file.parts or file.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
        ]
        manifest.append(f"{name}=={package.version}")
        manifest.extend(f"  runtime/Lib/site-packages/{path}" for path in sorted(paths))
        if name.lower() == "streamlit":
            manifest.extend(["  runtime/licenses/LICENSE.streamlit.txt", f"  Source: {STREAMLIT_LICENSE_URL}"])
    (destination / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def _verify_runtime(runtime: Path) -> None:
    check = """
import importlib.util
import pathlib
import struct
import sys

import docx, httpx, langgraph, numpy, openpyxl, pandas, pyarrow, pypdf, rank_bm25, sklearn, sqlite3, streamlit

root = pathlib.Path(sys.executable).resolve().parent
assert sys.version_info[:3] == (3, 11, 18) and struct.calcsize('P') == 8
assert pathlib.Path(sys.base_prefix).resolve() == root
assert pathlib.Path(streamlit.__file__).resolve().is_relative_to(root)
for name in ('chromadb', 'torch', 'sentence_transformers', 'transformers'):
    assert importlib.util.find_spec(name) is None, name
print('Portable Python, Streamlit, LangGraph and native dependencies imported successfully.')
"""
    subprocess.run([str(runtime / "python.exe"), "-I", "-B", "-c", check], check=True)


def _write_zip(staged: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for file in sorted(staged.rglob("*")):
            relative = file.relative_to(staged)
            if not file.is_file() or "__pycache__" in relative.parts or file.suffix == ".pyc":
                continue
            # uv's generated console launchers can embed the build-time interpreter path.
            if relative.parts[:4] == ("runtime", "Lib", "site-packages", "bin"):
                continue
            if relative.as_posix() == "runtime/Lib/site-packages/.lock":
                continue
            entry = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 11, 10, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            with file.open("rb") as source, archive.open(entry, "w") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-archive", type=Path, help="Use a previously downloaded CPython archive.")
    parser.add_argument("--streamlit-license", type=Path, help="Use the Streamlit 1.67.0 source LICENSE file.")
    parser.add_argument("--offline", action="store_true", help="Resolve wheel installs from the local uv cache.")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Build this release on x64 Windows with uv and the Windows tar executable available.")
    if args.offline and (args.python_archive is None or args.streamlit_license is None):
        parser.error("Offline builds require --python-archive and --streamlit-license.")

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="portable-build-", dir=RELEASE_DIR) as temp:
        work = Path(temp).resolve()
        if work.parent != RELEASE_DIR.resolve():
            raise ValueError("The temporary directory must be inside the release output directory.")
        source_archive = args.python_archive or _download(PYTHON_URL, work / PYTHON_ARCHIVE)
        if _sha256(source_archive) != PYTHON_SHA256:
            raise ValueError("CPython archive SHA-256 does not match the pinned upstream release.")
        license_file = args.streamlit_license or _download(STREAMLIT_LICENSE_URL, work / "LICENSE.streamlit.txt")
        if _sha256(license_file) != STREAMLIT_LICENSE_SHA256:
            raise ValueError("Streamlit LICENSE SHA-256 does not match the pinned upstream source.")

        subprocess.run(
            ["tar.exe", "-xf", str(source_archive.resolve()), "-C", str(work),
             "python/install", "python/licenses", "python/PYTHON.json"], check=True,
        )
        staged = work / "AI Interview Demo"
        _copy_project(staged)
        runtime = staged / "runtime"
        _copy_runtime(work / "python", runtime)
        site = _install_dependencies(runtime, args.offline)
        _write_notices(staged, site, license_file)
        _verify_runtime(runtime)

        temporary_zip = work / ZIP_NAME
        _write_zip(staged, temporary_zip)
        final_zip = RELEASE_DIR / ZIP_NAME
        temporary_zip.replace(final_zip)
    print(f"Built {final_zip} ({final_zip.stat().st_size:,} bytes)")
    print(f"SHA-256 {_sha256(final_zip)}")


if __name__ == "__main__":
    main()
