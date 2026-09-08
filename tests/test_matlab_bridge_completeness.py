"""Every MATLAB function in a bridged package is recorded, and each
recorded ``matlab_last_sync_hash`` matches the latest git commit that
touched the MATLAB file.

MATLAB counterpart: everything under ``/home/user/NDR-matlab/+ndr/**``
against every ``src/ndr/**/ndr_matlab_python_bridge.yaml`` in this repo.

Two properties held by this guard:

**Completeness.** Every ``.m`` file under a bridged MATLAB package has a
bridge entry. Two ways to satisfy an entry: a real port, or an entry with
a ``status`` from the vocabulary in section 6 of
``docs/developer_notes/ndr_matlab_python_bridge.yaml`` plus a
``decision_log`` saying why there is none. What is not fine is silence --
a MATLAB file that landed with no entry is exactly how NDR-python got
behind on the recent SmartSPIM / OME-Zarr additions.

That the recorded status is a LEGAL one, and that no file is recorded
twice, are checked next door in ``test_matlab_bridge_conventions.py``.

**Hash currency.** Every entry that carries a ``matlab_last_sync_hash``
must record the LATEST commit that touched its MATLAB file. When MATLAB
edits the file, the recorded hash goes stale and this test flags it, so
the Python port is reviewed against the change rather than silently
falling behind.

WHERE THE MATLAB TREE COMES FROM
This needs NDR-matlab checked out; see :func:`matlab_root`. Absent, the
check skips -- unless ``NDR_BRIDGE_CHECK_STRICT`` is set, which CI does
after checking the repo out, so a workflow that stops providing the tree
fails instead of quietly passing. Same reasoning as NDI-python's
``test_matlab_bridge_completeness.py``: a check that could not run must
not report the same result as a check that ran and passed.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

#: Set by CI once NDR-matlab is checked out.
STRICT_ENV_VAR = "NDR_BRIDGE_CHECK_STRICT"

#: Explicit override for the NDR-matlab checkout location (the repo root,
#: not ``+ndr``).
MATLAB_PATH_ENV_VAR = "NDR_MATLAB_PATH"

BRIDGE_FILENAME = "ndr_matlab_python_bridge.yaml"

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Extensions that are functions or apps. Everything else under a MATLAB
#: package (README.md, resource files) is not something a bridge entry
#: could describe.
FUNCTION_SUFFIXES = (".m", ".mlapp")


@dataclass(frozen=True)
class BridgedPackage:
    """A Python package and the MATLAB package it mirrors.

    Attributes:
        python_dir: Relative to the repo root. Every bridge YAML at or
            below it is read, so a package's own file and any bridge
            files in sub-packages all count as places an entry may live.
        matlab_dir: Relative to NDR-matlab's repo root (e.g.
            ``+ndr/+format/+omezarr``).
        excluded_layers: ``(path prefix, reason)`` for whole subtrees
            that are deliberately not bridged. A prefix, not a file list.
    """

    python_dir: str
    matlab_dir: str
    excluded_layers: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def id(self) -> str:
        return self.python_dir


#: The packages this guard covers. Adding one is a single entry.
#:
#: The MATLAB namespace has these top-level packages: ``+data``, ``+file``,
#: ``+format``, ``+fun``, ``+reader``, ``+string``, ``+time``. ``+docs``,
#: ``+setup``, and ``+test`` are MATLAB-only tooling; ``+setup`` has a
#: Python counterpart (``ndr.setup``) but its one Python module maps to
#: a MATLAB helper unrelated to the +setup MATLAB package layout, so the
#: package as a whole is excluded from the guard here.
PACKAGES = [
    BridgedPackage(python_dir="src/ndr/data", matlab_dir="+ndr/+data"),
    BridgedPackage(python_dir="src/ndr/file", matlab_dir="+ndr/+file"),
    BridgedPackage(python_dir="src/ndr/fun", matlab_dir="+ndr/+fun"),
    BridgedPackage(python_dir="src/ndr/reader", matlab_dir="+ndr/+reader"),
    BridgedPackage(python_dir="src/ndr/string", matlab_dir="+ndr/+string"),
    BridgedPackage(python_dir="src/ndr/time", matlab_dir="+ndr/+time"),
    # Format sub-packages: each has its own bridge YAML.
    BridgedPackage(python_dir="src/ndr/format/axon", matlab_dir="+ndr/+format/+axon"),
    BridgedPackage(
        python_dir="src/ndr/format/binarymatrix",
        matlab_dir="+ndr/+format/+binarymatrix",
    ),
    BridgedPackage(python_dir="src/ndr/format/bjg", matlab_dir="+ndr/+format/+bjg"),
    BridgedPackage(python_dir="src/ndr/format/ced", matlab_dir="+ndr/+format/+ced"),
    BridgedPackage(
        python_dir="src/ndr/format/dabrowska",
        matlab_dir="+ndr/+format/+dabrowska",
    ),
    BridgedPackage(python_dir="src/ndr/format/intan", matlab_dir="+ndr/+format/+intan"),
    BridgedPackage(python_dir="src/ndr/format/neo", matlab_dir="+ndr/+format/+neo"),
    BridgedPackage(
        python_dir="src/ndr/format/neuropixelsGLX",
        matlab_dir="+ndr/+format/+neuropixelsGLX",
    ),
    BridgedPackage(
        python_dir="src/ndr/format/omezarr",
        matlab_dir="+ndr/+format/+omezarr",
    ),
    BridgedPackage(
        python_dir="src/ndr/format/prairieview",
        matlab_dir="+ndr/+format/+prairieview",
    ),
    BridgedPackage(
        python_dir="src/ndr/format/smartspim",
        matlab_dir="+ndr/+format/+smartspim",
    ),
    BridgedPackage(
        python_dir="src/ndr/format/spikegadgets",
        matlab_dir="+ndr/+format/+spikegadgets",
    ),
    BridgedPackage(
        python_dir="src/ndr/format/stereoseq",
        matlab_dir="+ndr/+format/+stereoseq",
    ),
    BridgedPackage(python_dir="src/ndr/format/tdt", matlab_dir="+ndr/+format/+tdt"),
    BridgedPackage(
        python_dir="src/ndr/format/textSignal",
        matlab_dir="+ndr/+format/+textSignal",
    ),
    BridgedPackage(python_dir="src/ndr/format/vld", matlab_dir="+ndr/+format/+vld"),
    BridgedPackage(
        python_dir="src/ndr/format/whitematter",
        matlab_dir="+ndr/+format/+whitematter",
    ),
]


def matlab_root() -> Path | None:
    """The NDR-matlab repo root, or None if not checked out.

    ``$NDR_MATLAB_PATH`` when set, otherwise ``../NDR-matlab`` beside
    this checkout -- both the usual local layout and the one the CI
    workflow creates.
    """
    override = os.environ.get(MATLAB_PATH_ENV_VAR, "").strip()
    candidate = Path(override) if override else REPO_ROOT.parent / "NDR-matlab"
    return candidate if (candidate / "+ndr").is_dir() else None


def require_matlab_root() -> Path:
    """:func:`matlab_root`, skipping the test when absent -- unless strict."""
    root = matlab_root()
    if root is not None:
        return root
    message = (
        "NDR-matlab is not checked out, so bridge completeness cannot be "
        f"checked. Clone it beside this repo or set {MATLAB_PATH_ENV_VAR}."
    )
    if os.environ.get(STRICT_ENV_VAR, "").strip():
        pytest.fail(
            f"{message} ({STRICT_ENV_VAR} is set, so the tree was supposed to "
            "be there -- a missing one means the workflow stopped providing it, "
            "and skipping would report that as a pass.)"
        )
    pytest.skip(message)


def require_full_history(root: Path) -> None:
    """Fail loudly when the NDR-matlab checkout is shallow.

    ``actions/checkout`` is shallow by default, and a shallow clone
    collapses every file's history to the single commit it fetched:
    ``git log -- <path>`` then names that commit for EVERY file, so every
    recorded hash reads as stale, and ``git cat-file -t`` reports older
    commits as missing. That is hundreds of failures none of which is the
    actual problem. Say the actual problem once, first.

    CI avoids this with ``fetch-depth: 0``; locally, ``git fetch
    --unshallow``.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-shallow-repository"],
            check=True,
            capture_output=True,
            text=True,
        )
        shallow = result.stdout.strip() == "true"
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Older git without the flag: the marker file is the same signal.
        shallow = (root / ".git" / "shallow").exists()
    if shallow:
        pytest.fail(
            f"the NDR-matlab checkout at {root} is SHALLOW. Every file's history "
            "collapses to the commit it fetched, so `git log -- <path>` names the "
            "wrong latest commit for every entry and older commits read as "
            "missing -- the bridge hash checks would emit one false failure per "
            "entry. Run `git -C "
            f"{root} fetch --unshallow`, or give the checkout step `fetch-depth: 0`."
        )


# ---------------------------------------------------------------------------
# Reading a bridge file
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BridgeEntry:
    """One entry recording a MATLAB file, flattened for lookup.

    ``hash`` is the raw string as recorded in ``matlab_last_sync_hash``
    (may be short or long); ``source`` is the YAML file that defined it.
    """

    matlab_path: str
    hash: str | None
    source: Path


@dataclass(frozen=True)
class BridgeIndex:
    """What a package's bridge YAMLs record."""

    names: frozenset[str]
    paths: frozenset[str]
    entries: tuple[BridgeEntry, ...]
    sources: tuple[Path, ...]

    @property
    def path_stems(self) -> set[str]:
        return {Path(p).stem for p in self.paths if not p.endswith("/")}

    def records(self, matlab_path: str) -> bool:
        """True if this MATLAB file is recorded, by path or by name."""
        if matlab_path in self.paths:
            return True
        # A directory-shaped matlab_path covers everything under it.
        if any(matlab_path.startswith(p) for p in self.paths if p.endswith("/")):
            return True
        # A bare name vouches for a file only when NO entry claims that
        # stem by path. Otherwise a name recorded for one package
        # silently covers a same-named file in another.
        stem = Path(matlab_path).stem
        return stem in self.names and stem not in self.path_stems


def normalize_matlab_path(value: str) -> str:
    """A ``matlab_path:`` value as a path relative to NDR-matlab's root.

    Returns ``""`` for a value that names no file (``""``, ``"N/A"``).
    """
    normalized = value.replace("\\", "/").strip().lstrip("/")
    if not normalized or normalized == "N/A":
        return ""
    return normalized


def _collect(
    node: Any,
    names: set[str],
    paths: set[str],
    entries: list[BridgeEntry],
    source: Path,
    current_hash: str | None = None,
) -> None:
    """Walk a parsed YAML tree, gathering names, paths and hash entries.

    A hash inherits down the tree so a class-level ``matlab_last_sync_hash``
    still applies to its methods when they name their own ``matlab_path``
    without repeating the hash. An entry is emitted only where a
    ``matlab_path`` is DEFINED on the current node -- we don't emit one for
    every intermediate value in that path's subtree, because then a
    function with five input_arguments would look like six stale entries.
    """
    if isinstance(node, dict):
        local_hash = current_hash
        if isinstance(node.get("matlab_last_sync_hash"), str):
            local_hash = node["matlab_last_sync_hash"].strip()
        if isinstance(node.get("matlab_path"), str):
            normalized = normalize_matlab_path(node["matlab_path"])
            if normalized:
                paths.add(normalized)
                if local_hash and not normalized.endswith("/"):
                    entries.append(BridgeEntry(normalized, local_hash, source))
        for key, value in node.items():
            if key in ("name", "matlab_equivalent") and isinstance(value, str):
                names.add(value)
            _collect(value, names, paths, entries, source, local_hash)
    elif isinstance(node, list):
        for item in node:
            _collect(item, names, paths, entries, source, current_hash)


def read_bridge_index(package: BridgedPackage) -> BridgeIndex:
    """Read every bridge YAML at or below the package's Python directory."""
    sources = sorted((REPO_ROOT / package.python_dir).rglob(BRIDGE_FILENAME))
    assert sources, f"{package.python_dir} has no {BRIDGE_FILENAME}"
    names: set[str] = set()
    paths: set[str] = set()
    entries: list[BridgeEntry] = []
    for source in sources:
        _collect(yaml.safe_load(source.read_text(encoding="utf-8")), names, paths, entries, source)
    return BridgeIndex(frozenset(names), frozenset(paths), tuple(entries), tuple(sources))


def all_bridge_files() -> list[Path]:
    """Every bridge YAML anywhere in the repo, not just under a package."""
    return sorted((REPO_ROOT / "src").rglob(BRIDGE_FILENAME))


def matlab_functions(package: BridgedPackage, root: Path) -> list[str]:
    """Every function and app in the MATLAB package, as bridge-style paths."""
    base = root / package.matlab_dir
    assert base.is_dir(), f"{base} is not a directory"
    return sorted(
        f"{package.matlab_dir}/{path.relative_to(base).as_posix()}"
        for path in base.rglob("*")
        if path.suffix in FUNCTION_SUFFIXES
    )


def unrecorded(package: BridgedPackage, root: Path) -> list[str]:
    """MATLAB functions with no bridge entry and no layer exclusion."""
    index = read_bridge_index(package)
    excluded = tuple(prefix for prefix, _ in package.excluded_layers)
    return [
        path
        for path in matlab_functions(package, root)
        if not path.startswith(excluded) and not index.records(path)
    ]


# ---------------------------------------------------------------------------
# Git helpers for the hash-currency check
# ---------------------------------------------------------------------------


def _git_hash_for_path(root: Path, matlab_path: str) -> str | None:
    """Full hash of the latest commit that touched ``matlab_path``.

    Returns None when git doesn't know the file -- the completeness
    check catches missing files by a different route (and a hash entry
    naming a deleted file is caught by
    :meth:`test_every_recorded_matlab_path_points_at_a_real_file`).
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "log", "-n", "1", "--format=%H", "--", matlab_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    out = result.stdout.strip()
    return out or None


def _resolve_short_hash(root: Path, hash_value: str) -> str | None:
    """Resolve a short or long hash to its full 40-char form."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", hash_value],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    out = result.stdout.strip()
    return out if len(out) == 40 else None


# ---------------------------------------------------------------------------
# The guard: completeness
# ---------------------------------------------------------------------------


class TestTheBridgeFilesAreComplete:
    @pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.id)
    def test_every_matlab_function_is_recorded(self, package: BridgedPackage):
        """The regression guard.

        A MATLAB function that is neither recorded nor excluded fails
        here. Two ways to fix it, and both are fine: port it and add a
        ``functions:`` entry, or add an entry carrying a ``status`` from
        the documented vocabulary and a ``decision_log`` saying why not.
        What is not fine is silence -- which is how NDR-python got behind
        on the recent SmartSPIM and OME-Zarr additions.
        """
        missing = unrecorded(package, require_matlab_root())
        assert not missing, (
            f"{len(missing)} MATLAB function(s) under {package.matlab_dir} have no "
            f"entry in any {BRIDGE_FILENAME} below {package.python_dir}:\n  "
            + "\n  ".join(missing)
            + "\n\nAdd an entry recording the port, or one carrying a status "
            "from section 6 of docs/developer_notes/" + BRIDGE_FILENAME + " and a "
            "decision_log saying why there is none."
        )

    @pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.id)
    def test_every_recorded_matlab_path_points_at_a_real_file(self, package: BridgedPackage):
        """A renamed or removed MATLAB function leaves its entry pointing
        at nothing; the entry then looks maintained while describing a
        file that does not exist.
        """
        root = require_matlab_root()
        index = read_bridge_index(package)
        prefix = f"{package.matlab_dir}/"
        stale = sorted(
            path for path in index.paths if path.startswith(prefix) and not (root / path).exists()
        )
        assert not stale, (
            f"{BRIDGE_FILENAME} entries under {package.python_dir} name MATLAB "
            "files that do not exist:\n  "
            + "\n  ".join(stale)
            + '\n\nPoint each at its new path, or set matlab_path: "N/A" and say '
            "in the decision_log that MATLAB removed it."
        )

    @pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.id)
    def test_every_excluded_layer_still_exists(self, package: BridgedPackage):
        """An exclusion for a directory MATLAB no longer has is stale."""
        root = require_matlab_root()
        for prefix, reason in package.excluded_layers:
            assert (root / prefix).is_dir(), (
                f"excluded layer {prefix!r} is not a directory in NDR-matlab. "
                f"The exclusion's reason was: {reason}"
            )

    @pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.id)
    def test_every_excluded_layer_carries_a_reason(self, package: BridgedPackage):
        """An exclusion without a reason is indistinguishable from an omission."""
        for prefix, reason in package.excluded_layers:
            assert len(reason.split()) >= 10, f"exclusion {prefix!r} needs a real reason"


#: ``regular_port`` is the ordinary case and needs no justification --
#: mirroring MATLAB is what the port does by default, so there is no decision
#: to record, and demanding one on all 105 would produce boilerplate that
#: devalues the logs carrying real reasoning. Every OTHER status records a
#: divergence, and a divergence without a reason is what this check exists to
#: catch. ``test_matlab_bridge_conventions.py`` pins its copy of this set
#: against ours so the two files cannot disagree.
STATUSES_NEEDING_NO_DECISION_LOG = frozenset({"regular_port"})


class TestTheDeferralsSayWhy:
    """A divergence is a decision, not a label.

    An entry that records a divergent status but no reason passes the
    completeness check while telling the next reader nothing -- so the
    gap is "recorded" and still gets re-investigated. Same rule as
    NDI-python's guard.
    """

    @pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.id)
    def test_every_divergence_has_a_decision_log(self, package: BridgedPackage):
        undocumented = []
        for source in sorted((REPO_ROOT / package.python_dir).rglob(BRIDGE_FILENAME)):
            data = yaml.safe_load(source.read_text(encoding="utf-8"))
            for entry in _entries_with_status(data):
                if entry["status"].strip() in STATUSES_NEEDING_NO_DECISION_LOG:
                    continue
                if len((entry.get("decision_log") or "").split()) < 5:
                    name = entry.get("name", "<unnamed>")
                    undocumented.append(f"{source.relative_to(REPO_ROOT)}: {name}")
        assert not undocumented, (
            "bridge entries that record a divergence but no decision_log "
            "explaining it:\n  " + "\n  ".join(undocumented)
        )


def _entries_with_status(node: Any) -> list[dict[str, Any]]:
    """Every mapping in the tree that carries a ``status:`` key."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if isinstance(node.get("status"), str):
            found.append(node)
        for value in node.values():
            found.extend(_entries_with_status(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_entries_with_status(item))
    return found


# ---------------------------------------------------------------------------
# The guard: hash currency
# ---------------------------------------------------------------------------


def _collect_hash_entries() -> list[BridgeEntry]:
    """Every entry across every bridge yaml that carries a hash."""
    entries: list[BridgeEntry] = []
    for source in all_bridge_files():
        names: set[str] = set()
        paths: set[str] = set()
        _collect(
            yaml.safe_load(source.read_text(encoding="utf-8")),
            names,
            paths,
            entries,
            source,
        )
    return entries


def _commits_touching_since(root: Path, matlab_path: str, since: str) -> list[str] | None:
    """Commits touching ``matlab_path`` after ``since``, newest first.

    This is the drift question, and it is PATH-FILTERED: ``-- <matlab_path>``
    restricts the walk to commits touching that one file, so unrelated
    activity in NDR-matlab never makes an entry drift no matter what form the
    recorded hash takes.

    Returns None when git cannot answer -- an unresolvable hash, say. That is
    not silently a pass: the caller reports it, and
    ``test_matlab_bridge_conventions.py`` names the bad object precisely.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "log",
                "--format=%h",
                f"{since}..HEAD",
                "--",
                matlab_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.split()


class TestNoEntryHasDrifted:
    """An entry DRIFTS when NDR-matlab has commits touching its
    ``matlab_path`` after the recorded hash. Drift fails CI.

        git log <matlab_last_sync_hash>..HEAD -- <matlab_path>

    Set for all three bridge repos by NDI-python#211. The rule is
    file-scoped: only commits touching THIS file count, so unrelated
    NDR-matlab activity never trips it.

    WHY NOT "the hash equals the file's last-touching commit". Because that
    asks about the hash's FORM rather than about the file, and rejects
    records that are true. Both forms are legal, and drift is sound under
    both:

    * the file's own last-touching commit -- the documented default, and the
      more informative of the two: "I examined this version of this file";
    * a repo-wide commit, e.g. NDR-matlab HEAD when a batch was examined
      together -- a broader claim, "I examined as of this repo state", and
      still sound, since any later change to the file is caught and changes
      before it fall inside what was examined.

    Equals-latest failed the second form on entries whose records were TRUE,
    and the only way to clear the resulting red build was to rewrite a
    correct record with a different, equally correct value. NDI-python had
    103 such entries. That is how a check teaches contributors to ignore it.

    When this test DOES fail, the file really has changed since it was
    examined. Two ways to clear it, both fine: review the MATLAB diff and
    port the change, or leave the Python alone and bump the hash with a
    ``decision_log`` note saying why the change is a no-op here.
    """

    def test_no_matlab_file_changed_after_its_recorded_hash(self):
        root = require_matlab_root()
        require_full_history(root)
        entries = _collect_hash_entries()
        assert entries, "no matlab_last_sync_hash entries found"

        drifted: list[str] = []
        unresolvable: list[str] = []
        for entry in entries:
            if _git_hash_for_path(root, entry.matlab_path) is None:
                # git doesn't know the file; the stale-path test covers it.
                continue
            since = _commits_touching_since(root, entry.matlab_path, entry.hash)
            src_rel = entry.source.relative_to(REPO_ROOT)
            if since is None:
                unresolvable.append(f"{src_rel}: {entry.matlab_path}  recorded: {entry.hash}")
                continue
            if since:
                shown = ", ".join(since[:5]) + (", ..." if len(since) > 5 else "")
                drifted.append(
                    f"{src_rel}: {entry.matlab_path}\n"
                    f"      recorded: {entry.hash}   "
                    f"commits touching it since: {len(since)} ({shown})"
                )

        assert not unresolvable, (
            "git could not walk history from these recorded hashes -- they do "
            "not name anything in NDR-matlab:\n  "
            + "\n  ".join(unresolvable)
            + "\n\nSee test_matlab_bridge_conventions.py, which names the bad "
            "object type."
        )
        assert not drifted, (
            f"{len(drifted)} bridge entr{'y has' if len(drifted) == 1 else 'ies have'} "
            "DRIFTED -- NDR-matlab has commits touching the MATLAB file after the "
            "recorded matlab_last_sync_hash:\n\n  "
            + "\n  ".join(drifted)
            + "\n\nReview the MATLAB diff between the recorded hash and HEAD, port "
            "any behavioral changes to Python, and update matlab_last_sync_hash. If "
            "nothing needs to change on the Python side, bump the hash and add a "
            "short note in the decision_log explaining why the MATLAB change is a "
            "no-op here (e.g. comment-only, MATLAB-analyzer fix)."
        )


# ---------------------------------------------------------------------------
# Self-tests: the guard can actually fail
# ---------------------------------------------------------------------------


class TestTheGuardWouldActuallyCatchOne:
    """A guard that cannot fail is worse than none. These prove it can."""

    def test_an_unrecorded_function_is_reported(self):
        index = BridgeIndex(frozenset({"readGEF"}), frozenset(), (), ())
        assert index.records("+ndr/+format/+stereoseq/readGEF.m")
        assert not index.records("+ndr/+format/+stereoseq/helloMatlab.m")

    def test_a_matlab_path_entry_counts(self):
        index = BridgeIndex(
            frozenset(),
            frozenset({"+ndr/+format/+stereoseq/helloMatlab.m"}),
            (),
            (),
        )
        assert index.records("+ndr/+format/+stereoseq/helloMatlab.m")

    def test_na_is_not_a_reference_to_anything(self):
        names: set[str] = set()
        paths: set[str] = set()
        entries: list[BridgeEntry] = []
        _collect({"matlab_path": "N/A"}, names, paths, entries, Path("x.yaml"))
        assert paths == set()
        assert entries == []

    def test_a_directory_matlab_path_covers_the_package_under_it(self):
        index = BridgeIndex(
            frozenset(),
            frozenset({"+ndr/+format/+omezarr/private/"}),
            (),
            (),
        )
        assert index.records("+ndr/+format/+omezarr/private/parseDType.m")
        assert not index.records("+ndr/+format/+omezarr/readArray.m")

    @pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.id)
    def test_a_new_matlab_function_in_any_covered_package_would_fail(self, package):
        """Every registered package really is guarded, not merely listed."""
        index = read_bridge_index(package)
        assert not index.records(f"{package.matlab_dir}/aFunctionNobodyHasWritten.m")


class TestDriftIsFileScopedAndAcceptsBothHashForms:
    """The drift rule, exercised against real NDR-matlab history.

    Fixtures are derived from the checkout rather than hardcoded, so these
    keep testing the mechanism as MATLAB history moves on.
    """

    def _a_file_with_history(self, root: Path) -> tuple[str, list[str]]:
        """A bridged MATLAB file and its commits, newest first (>= 2)."""
        for entry in _collect_hash_entries():
            commits = subprocess.run(
                ["git", "-C", str(root), "log", "--format=%h", "--", entry.matlab_path],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()
            if len(commits) >= 2:
                return entry.matlab_path, commits
        pytest.skip("no bridged MATLAB file has two commits to compare")

    def test_an_older_hash_drifts(self):
        """The case the guard exists for: the file moved after it was
        examined, so the record is no longer true."""
        root = require_matlab_root()
        require_full_history(root)
        path, commits = self._a_file_with_history(root)
        since = _commits_touching_since(root, path, commits[1])
        assert since, f"{path} should report drift from its second-newest commit"
        assert commits[0] in since

    def test_the_files_own_last_touching_commit_does_not_drift(self):
        """Form one of a legal hash: "I examined this version of this file"."""
        root = require_matlab_root()
        require_full_history(root)
        path, commits = self._a_file_with_history(root)
        assert _commits_touching_since(root, path, commits[0]) == []

    def test_a_repo_wide_commit_does_not_drift(self):
        """Form two: "I examined as of this repo state" -- a batch sync.

        This is the case equals-latest got wrong. NDR-matlab HEAD is not the
        file's own last-touching commit, so equals-latest would reject it;
        drift accepts it, because nothing has touched the file since.
        NDI-python had 103 entries of exactly this shape, all of them true.
        """
        root = require_matlab_root()
        require_full_history(root)
        path, commits = self._a_file_with_history(root)
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Precondition: this really is the divergent case, or the test proves
        # nothing -- HEAD must NOT be the file's own last-touching commit.
        if head == commits[0]:
            pytest.skip("HEAD is this file's last-touching commit; no divergence to test")

        assert (
            _commits_touching_since(root, path, head) == []
        ), "a repo-wide hash must not drift when nothing has touched the file since"

    def test_unrelated_repo_activity_does_not_drift_an_entry(self):
        """File-scoped, not repo-scoped: commits that do not touch the file
        never make it drift. This is the property @stevevanhooser asked for --
        "a change is only noticed when the file changes rather than anything
        on the repo"."""
        root = require_matlab_root()
        require_full_history(root)
        path, commits = self._a_file_with_history(root)
        total_since = subprocess.run(
            ["git", "-C", str(root), "log", "--format=%h", f"{commits[0]}..HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        touching_since = _commits_touching_since(root, path, commits[0])
        assert touching_since == []
        # The repo moved on; the entry did not drift. That is the whole point.
        if not total_since:
            pytest.skip("no unrelated commits after this file's last change")
        assert len(total_since) > len(touching_since)


class TestHashCollectionCollectsNestedEntries:
    """A hash sitting under a top-level entry inherits to its nested entries.

    Reader classes in NDR-python's bridge tend to record one hash on the
    class and then list methods without their own hash. A collector that
    stops at the top level misses those.
    """

    def test_a_hash_at_the_top_of_a_class_covers_its_matlab_path(self):
        entries: list[BridgeEntry] = []
        names: set[str] = set()
        paths: set[str] = set()
        _collect(
            {
                "classes": [
                    {
                        "name": "intan_rhd",
                        "matlab_path": "+ndr/+reader/intan_rhd.m",
                        "matlab_last_sync_hash": "20743f7",
                        "methods": [{"name": "read"}],
                    }
                ]
            },
            names,
            paths,
            entries,
            Path("x.yaml"),
        )
        assert len(entries) == 1
        assert entries[0].matlab_path == "+ndr/+reader/intan_rhd.m"
        assert entries[0].hash == "20743f7"


if __name__ == "__main__":
    pytest.main([__file__])
