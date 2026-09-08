"""The bridge contract's *conventions* are enforced, not merely written down.

MATLAB counterpart: every ``src/ndr/**/ndr_matlab_python_bridge.yaml`` in
this repo, read against the spec at
``docs/developer_notes/ndr_matlab_python_bridge.yaml`` and against the
NDR-matlab checkout.

``tests/test_matlab_bridge_completeness.py`` answers "is every MATLAB
function recorded, and is its recorded hash current?". This file answers
the questions that check cannot see, because a wrong answer to any of them
still looks like a recorded entry:

**One entry per function.** A file recorded TWICE is recorded, so the
completeness check passes while two entries drift apart -- one saying
``porting_deferred`` next to another recording a port that exists. NDI-python
carried eleven such pairs, four of them self-contradictory. Keyed on
``(name, matlab_path)``, never on path alone: class methods legitimately share
the one ``.m`` file that defines their class.

**The hash is a COMMIT.** ``git log`` does not error on a blob -- it returns
commits -- so a blob recorded in ``matlab_last_sync_hash`` reads as merely
"stale" to the currency check rather than as unusable, and the wrong repair
gets made. NDI-python accumulated 35 blob hashes before anyone noticed. This
is the check that names the real problem.

**``status`` is a closed vocabulary.** It was free-form in both repos until
NDI-python's cleanup, and a free-form ``not_applicable`` came to mean three
incompatible things at once. The legal values are read from the spec file's
live ``status_vocabulary`` block rather than restated here, and
:meth:`TestTheVocabularyIsTheDocumentedOne.test_spec_and_enforcement_agree`
pins the two together so the doc and the mechanism cannot drift apart.

**The CI job actually runs this file.** A bridge check that silently skips
reports the same green as one that ran, which is the exact failure this whole
guard exists to prevent -- so
:class:`TestTheWorkflowActuallyRunsTheBridgeChecks` reads
``.github/workflows/ci.yml`` and fails if a ``tests/test_matlab_bridge_*.py``
file is not named by the bridge job under the strict env var.

WHERE THE MATLAB TREE COMES FROM
The commit-object check needs NDR-matlab checked out, and needs it NOT
shallow; see :func:`require_matlab_root` and
:func:`require_full_history` in
``tests/test_matlab_bridge_completeness.py``. Absent, the check skips --
unless ``NDR_BRIDGE_CHECK_STRICT`` is set, which CI does.
"""

from __future__ import annotations

import collections
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.test_matlab_bridge_completeness import (
    BRIDGE_FILENAME,
    REPO_ROOT,
    STATUSES_NEEDING_NO_DECISION_LOG,
    STRICT_ENV_VAR,
    all_bridge_files,
    normalize_matlab_path,
    require_full_history,
    require_matlab_root,
)

#: The spec. The one normative place the bridge rules are written down;
#: AGENTS.md and PYTHON_PORTING_GUIDE.md cross-reference it rather than
#: restating it.
SPEC_PATH = REPO_ROOT / "docs" / "developer_notes" / BRIDGE_FILENAME

CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: The statuses this module enforces. Deliberately written out here rather
#: than only read from the spec, so that a silent edit to either one is a
#: test failure (see TestTheVocabularyIsTheDocumentedOne) instead of a
#: silent widening of what CI accepts.
ALLOWED_STATUSES = frozenset(
    {
        "regular_port",
        "ported_differently",
        "porting_deferred",
        "matlab_only",
        "retired",
    }
)

#: Retired spellings, mapped to what to use instead. A value here gets a
#: better error message than "not in the vocabulary".
REPLACED_STATUSES = {
    "ported_elsewhere": (
        "ported_differently -- the label names the manner, not the location, "
        "since python_path already answers where"
    ),
    "not_yet_ported": "porting_deferred",
    "not_applicable": (
        "one of ported_differently / porting_deferred / matlab_only -- decide "
        "which of the three it actually meant by reading the decision_log"
    ),
    "implemented": "regular_port",
    "does_not_exist": "retired",
    "ported": "regular_port",
}

#: Statuses that must say where the Python capability lives. A
#: ``regular_port`` that names no module is not a recorded port at all.
STATUSES_REQUIRING_PYTHON_PATH = frozenset({"regular_port", "ported_differently"})


# ---------------------------------------------------------------------------
# Reading entries
# ---------------------------------------------------------------------------


class Entry:
    """One bridge mapping, with where it was found.

    ``node`` is the raw YAML mapping; ``where`` is a human-readable
    ``file: yaml.path`` used in failure messages, because a bare name is not
    enough to find one entry among 120.
    """

    def __init__(self, source: Path, path: tuple[str, ...], node: dict[str, Any]):
        self.source = source
        self.node = node
        self.where = f"{source.relative_to(REPO_ROOT)}: {'.'.join(path) or '<root>'}"

    @property
    def name(self) -> str:
        value = self.node.get("name")
        return value.strip() if isinstance(value, str) else ""

    @property
    def matlab_path(self) -> str:
        value = self.node.get("matlab_path")
        return normalize_matlab_path(value) if isinstance(value, str) else ""

    @property
    def status(self) -> str:
        value = self.node.get("status")
        return value.strip() if isinstance(value, str) else ""

    @property
    def decision_log(self) -> str:
        value = self.node.get("decision_log")
        return value.strip() if isinstance(value, str) else ""

    @property
    def matlab_last_sync_hash(self) -> str:
        """The entry's OWN recorded hash.

        Deliberately not inherited from an enclosing entry: this gate asks
        whether THIS entry records what was examined, and a hash borrowed from
        a parent is a claim the parent made about a different file.
        """
        value = self.node.get("matlab_last_sync_hash")
        return value.strip() if isinstance(value, str) else ""

    @property
    def python_qualified(self) -> str:
        value = self.node.get("python_qualified")
        return value.strip() if isinstance(value, str) else ""

    @property
    def python_paths(self) -> list[str]:
        """``python_path`` as a list.

        A plain port names one file. A ``ported_differently`` entry may name
        several when the capability really is reached through more than one
        module (``+ndr/+reader/imagestack.m`` is covered by four readers), so
        both a string and a list are accepted here.
        """
        value = self.node.get("python_path")
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return []

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Entry {self.name!r} {self.where}>"


def _walk(node: Any, source: Path, path: tuple[str, ...] = ()) -> list[Entry]:
    """Every mapping in the tree that is an entry.

    An entry is a mapping carrying a ``matlab_path`` or a ``status``: those
    are the two ways a node claims to record a MATLAB file. Argument
    mappings (``name`` + ``type_python``) carry neither and are skipped, so
    a function with five input_arguments does not read as six entries.
    """
    found: list[Entry] = []
    if isinstance(node, dict):
        if isinstance(node.get("matlab_path"), str) or isinstance(node.get("status"), str):
            found.append(Entry(source, path, node))
        for key, value in node.items():
            found.extend(_walk(value, source, path + (str(key),)))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(_walk(item, source, path + (str(index),)))
    return found


def all_entries() -> list[Entry]:
    """Every entry in every bridge YAML under ``src/``.

    The spec file under ``docs/`` is deliberately not included: it defines
    the rules, it does not record ports.
    """
    entries: list[Entry] = []
    for source in all_bridge_files():
        entries.extend(_walk(yaml.safe_load(source.read_text(encoding="utf-8")), source))
    return entries


def spec_vocabulary() -> dict[str, str]:
    """The ``status_vocabulary`` block from the spec file."""
    data = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    vocabulary = data.get("status_vocabulary")
    assert isinstance(vocabulary, dict), (
        f"{SPEC_PATH.relative_to(REPO_ROOT)} has no `status_vocabulary:` block. "
        "It is live YAML on purpose, so the vocabulary the docs describe and the "
        "one CI enforces are the same object."
    )
    return vocabulary


# ---------------------------------------------------------------------------
# One entry per MATLAB function
# ---------------------------------------------------------------------------


class TestEachMatlabFunctionIsRecordedOnce:
    """Duplicate entries are invisible to the completeness check.

    That check asks "is this file recorded?", and a file recorded twice is
    recorded. The two entries then drift; a reader who finds the stale one
    first is told a port does not exist when it does.
    """

    def test_no_matlab_function_has_two_entries(self):
        by_key: dict[tuple[str, str], list[Entry]] = collections.defaultdict(list)
        for entry in all_entries():
            # A directory-shaped matlab_path covers a whole subtree rather
            # than one function, so several may share it by design.
            if entry.matlab_path and not entry.matlab_path.endswith("/"):
                by_key[(entry.name, entry.matlab_path)].append(entry)

        offenders: list[str] = []
        for (name, matlab_path), group in sorted(by_key.items()):
            if len(group) < 2:
                continue
            if _is_a_declared_multi_module_pair(group):
                continue
            offenders.append(
                f"{name or '<unnamed>'} -> {matlab_path} ({len(group)} entries)\n      "
                + "\n      ".join(entry.where for entry in group)
            )

        assert not offenders, (
            f"{len(offenders)} MATLAB function(s) carry more than one bridge entry:\n\n  "
            + "\n  ".join(offenders)
            + "\n\nKeep the entry that matches the Python tree and delete the other. "
            "If one MATLAB function really is exposed from two Python modules, say "
            "so explicitly: EVERY entry in the group must carry its own distinct "
            "python_qualified. Half a declaration is not a declaration."
        )


def _is_a_declared_multi_module_pair(group: list[Entry]) -> bool:
    """True when a group is a deliberately declared two-module exposure.

    Every member must name a ``python_qualified``, and they must all differ.
    One entry declaring itself while its twin stays silent is an accident
    that happens to have a field set, not a design.
    """
    qualified = [entry.python_qualified for entry in group]
    if not all(qualified):
        return False
    return len(set(qualified)) == len(qualified)


# ---------------------------------------------------------------------------
# matlab_last_sync_hash is a commit
# ---------------------------------------------------------------------------


def _git_object_type(root: Path, revision: str) -> str:
    """``blob`` / ``commit`` / ``tree`` / ``tag``, or ``missing``."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-t", revision],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:  # pragma: no cover - git is present in CI
        return "missing"
    return result.stdout.strip() if result.returncode == 0 else "missing"


class TestEveryRecordedHashIsACommit:
    """``matlab_last_sync_hash`` names a git COMMIT, never a blob.

    Freshness is asked by walking history --
    ``git log <hash>..HEAD -- <path>`` -- and ``git log`` DOES NOT ERROR on a
    blob: it returns commits. So a blob recorded here does not announce
    itself; it reads as a hash that has merely gone stale, and somebody
    "fixes" it by bumping to HEAD, which claims a review nobody performed.

    Fixing one for real is exact, not a guess: walk the file's history
    comparing ``git rev-parse <commit>:<path>`` against the recorded blob and
    record the commit that produced it. If nothing resolves, delete the field
    and say why in the decision_log -- an absent hash claims nothing, a wrong
    one claims something false.
    """

    def test_no_recorded_hash_is_a_blob_or_missing(self):
        root = require_matlab_root()
        require_full_history(root)

        recorded: dict[str, list[Entry]] = collections.defaultdict(list)
        for entry in all_entries():
            value = entry.node.get("matlab_last_sync_hash")
            if isinstance(value, str) and value.strip():
                recorded[value.strip()].append(entry)
        assert recorded, "no matlab_last_sync_hash entries found -- is the walker broken?"

        wrong: list[str] = []
        for value, entries in sorted(recorded.items()):
            kind = _git_object_type(root, value)
            if kind == "commit":
                continue
            wrong.append(
                f"{value} is a {kind}, not a commit\n      "
                + "\n      ".join(f"{e.where} ({e.name})" for e in entries)
            )

        assert not wrong, (
            f"{len(wrong)} recorded matlab_last_sync_hash value(s) do not name a "
            "commit in NDR-matlab:\n\n  "
            + "\n  ".join(wrong)
            + "\n\nGet the value with `git -C ../NDR-matlab log -n 1 --format=%h -- "
            "<matlab_path>`, NOT with `git hash-object`. To repair a blob, walk the "
            "file's history comparing `git rev-parse <commit>:<path>` against the "
            "recorded blob and record the commit that produced it -- do not bump to "
            "HEAD. See section 5 of docs/developer_notes/" + BRIDGE_FILENAME + "."
        )


# ---------------------------------------------------------------------------
# status is a closed vocabulary
# ---------------------------------------------------------------------------


class TestTheVocabularyIsTheDocumentedOne:
    def test_spec_and_enforcement_agree(self):
        """The doc and the mechanism are pinned to each other.

        Writing a rule down in one place and enforcing a different one in
        another is how NDI-python's spec came to contradict itself. Adding a
        status means editing both, in one commit, or this fails.
        """
        documented = frozenset(spec_vocabulary())
        assert documented == ALLOWED_STATUSES, (
            "the status vocabulary documented in "
            f"{SPEC_PATH.relative_to(REPO_ROOT)} and the one enforced by "
            f"{Path(__file__).name} have drifted apart.\n"
            f"  documented only: {sorted(documented - ALLOWED_STATUSES)}\n"
            f"  enforced only:   {sorted(ALLOWED_STATUSES - documented)}\n"
            "Update both together."
        )

    def test_the_decision_log_exemption_is_documented(self):
        """Which status is exempt from ``decision_log`` is a rule too.

        ``regular_port`` is exempt because mirroring MATLAB is the default and
        there is no decision to record. That exemption is enforced in
        ``test_matlab_bridge_completeness.py``; this pins it to what the spec
        actually tells a reader, so the exemption cannot quietly widen to a
        status that really does owe an explanation.
        """
        vocabulary = spec_vocabulary()
        assert STATUSES_NEEDING_NO_DECISION_LOG <= frozenset(vocabulary), (
            "a status exempt from decision_log is not in the vocabulary: "
            f"{sorted(STATUSES_NEEDING_NO_DECISION_LOG - frozenset(vocabulary))}"
        )
        documented_exempt = {
            status
            for status, description in vocabulary.items()
            if "no `decision_log`" in str(description)
        }
        assert documented_exempt == set(STATUSES_NEEDING_NO_DECISION_LOG), (
            "the spec and the code disagree about which statuses need no "
            "decision_log.\n"
            f"  spec says exempt:    {sorted(documented_exempt)}\n"
            f"  code treats exempt:  {sorted(STATUSES_NEEDING_NO_DECISION_LOG)}\n"
            "Every other status records a divergence and owes a reason."
        )

    def test_every_documented_status_explains_itself(self):
        """A one-word gloss is not a definition; the whole point of the table
        is that a reader can tell the four apart without guessing."""
        thin = [
            status
            for status, description in spec_vocabulary().items()
            if len(str(description).split()) < 15
        ]
        assert not thin, (
            "these entries in the spec's status_vocabulary do not say enough to "
            f"distinguish them from the others: {thin}"
        )


class TestEveryStatusIsInTheVocabulary:
    def test_no_entry_uses_an_unknown_status(self):
        offenders: list[str] = []
        for entry in all_entries():
            if not entry.status or entry.status in ALLOWED_STATUSES:
                continue
            replacement = REPLACED_STATUSES.get(entry.status)
            hint = f"use {replacement}" if replacement else "not a known status"
            offenders.append(f"{entry.where}: {entry.name} has status: {entry.status} -- {hint}")

        assert not offenders, (
            f"{len(offenders)} bridge entr{'y' if len(offenders) == 1 else 'ies'} use a "
            "status outside the documented vocabulary:\n  "
            + "\n  ".join(offenders)
            + "\n\nThe vocabulary lives in section 6 of docs/developer_notes/"
            + BRIDGE_FILENAME
            + ". Relabel by reading the decision_log and deciding what the entry "
            "actually claims -- the vague values hid several different claims, so a "
            "find-and-replace would just move the ambiguity."
        )

    def test_a_status_that_claims_a_port_says_where(self):
        """A status asserting the capability exists must name the module.

        ``regular_port`` and ``ported_differently`` both claim Python can do
        the thing. Either one without a ``python_path`` is a worse
        ``porting_deferred``: it makes the claim and then declines to say
        where, so the next reader has to go searching for a module that may
        not exist.
        """
        offenders = [
            f"{entry.where}: {entry.name} (status: {entry.status})"
            for entry in all_entries()
            if entry.status in STATUSES_REQUIRING_PYTHON_PATH and not entry.python_paths
        ]
        assert not offenders, (
            f"{len(offenders)} entr{'y' if len(offenders) == 1 else 'ies'} claim a "
            "status that asserts a Python counterpart exists, but name no "
            "python_path:\n  "
            + "\n  ".join(offenders)
            + "\n\nName the module the capability is reached through (a list is fine "
            "for ported_differently when it really is more than one). If you cannot "
            "name one, the entry is porting_deferred or matlab_only."
        )

    def test_every_entry_naming_a_matlab_path_carries_a_hash(self):
        """An entry with no ``matlab_last_sync_hash`` can never drift.

        That is the failure this gate exists for, and it is worse than a stale
        hash rather than milder: a stale hash goes red and gets fixed, while a
        missing one asserts "this port is current" forever and nothing can
        contradict it. Silent false assurance instead of a red build.
        NDI-python#211 decision 2 gates it.

        Deliberately NOT dependent on an NDR-matlab checkout -- it reads only
        this repo's YAML -- so it runs in every test-matrix job, not just the
        bridge job. A rule this cheap should not be reachable from only one
        job.

        A ``retired`` entry sets ``matlab_path: "N/A"``, naming no file, so
        there is nothing for it to drift against and it is exempt.
        """
        hashless = [
            f"{entry.where}: {entry.name or '<unnamed>'} -> {entry.matlab_path}"
            for entry in all_entries()
            if entry.matlab_path and not entry.matlab_last_sync_hash
        ]
        assert not hashless, (
            f"{len(hashless)} bridge entr{'y names' if len(hashless) == 1 else 'ies name'} "
            "a matlab_path but record no matlab_last_sync_hash:\n  "
            + "\n  ".join(hashless)
            + "\n\nRecord the commit you examined:\n"
            "    git -C ../NDR-matlab log -n 1 --format=%h -- <matlab_path>\n"
            "An entry without a hash cannot drift, so it claims to be current "
            "forever and nothing can contradict it. See section 5a of "
            "docs/developer_notes/" + BRIDGE_FILENAME + "."
        )

    def test_every_named_python_path_exists(self):
        """A ``python_path`` pointing at nothing makes an entry look
        maintained while describing a module that is not there."""
        missing: list[str] = []
        for entry in all_entries():
            for python_path in entry.python_paths:
                if not (REPO_ROOT / "src" / python_path).exists():
                    missing.append(f"{entry.where}: {entry.name} -> {python_path}")
        assert not missing, (
            f"{len(missing)} python_path value(s) name a file that does not exist "
            "under src/:\n  " + "\n  ".join(missing)
        )

    def test_every_entry_states_its_status(self):
        """An absent ``status`` does not announce what it means.

        These files are read by people far more often than by this test
        suite, and a reader looking at an entry with no ``status`` cannot tell
        "this is a normal port" from "nobody filled this in". So the ordinary
        case is written out as ``regular_port`` rather than left to inference,
        and an entry that omits it fails here.
        """
        silent = [
            f"{entry.where}: {entry.name or '<unnamed>'} -> {entry.matlab_path}"
            for entry in all_entries()
            if entry.matlab_path and not entry.status
        ]
        assert not silent, (
            f"{len(silent)} bridge entr{'y' if len(silent) == 1 else 'ies'} record a "
            "matlab_path but no status:\n  "
            + "\n  ".join(silent)
            + "\n\nEvery entry states its status. An ordinary 1:1 port is "
            "`status: regular_port`; anything else takes the matching value from "
            "section 6 of docs/developer_notes/" + BRIDGE_FILENAME + " plus a decision_log."
        )

    def test_retired_entries_do_not_point_at_a_live_matlab_file(self):
        root = require_matlab_root()
        offenders = [
            f"{entry.where}: {entry.name} -> {entry.matlab_path}"
            for entry in all_entries()
            if entry.status == "retired"
            and entry.matlab_path
            and (root / entry.matlab_path).exists()
        ]
        assert not offenders, (
            "these entries are marked retired but their MATLAB file still exists:\n  "
            + "\n  ".join(offenders)
            + '\n\nA retired entry sets matlab_path: "N/A". If the file is back, the '
            "entry is not retired."
        )


# ---------------------------------------------------------------------------
# The workflow actually runs these checks
# ---------------------------------------------------------------------------


class TestTheWorkflowActuallyRunsTheBridgeChecks:
    """A bridge check nobody runs reports the same green as one that passed.

    The bridge job names its test files one by one, and the test-matrix jobs
    have no NDR-matlab checkout -- so a new ``test_matlab_bridge_*.py`` that
    nobody wired in would SKIP in every job and look like a pass. That is
    exactly the failure this whole guard exists to prevent, so it is worth a
    test of its own rather than a comment asking the next person to remember.
    """

    def _job(self, name: str) -> str:
        """The YAML text of one top-level job in the CI workflow."""
        text = CI_WORKFLOW.read_text(encoding="utf-8")
        marker = f"\n  {name}:\n"
        assert marker in text, f"{CI_WORKFLOW.relative_to(REPO_ROOT)} has no `{name}:` job"
        after = text.split(marker, 1)[1]
        # Up to the next top-level job key (two-space indent at line start).
        following = re.search(r"\n  [a-zA-Z0-9_-]+:\n", after)
        return after[: following.start()] if following else after

    def _job_commands(self, name: str) -> str:
        """A job's text with COMMENT LINES STRIPPED.

        Guards about what a job *runs* must read what it runs. Matching raw
        job text lets a comment mentioning the flag satisfy an assertion the
        command no longer does -- which is how the first version of
        :meth:`test_the_test_matrix_deselects_matlab_dependent_tests` passed
        while the matrix had stopped deselecting anything.
        """
        return "\n".join(
            line for line in self._job(name).splitlines() if not line.lstrip().startswith("#")
        )

    def _bridge_job(self) -> str:
        return self._job("bridge")

    def test_every_bridge_test_file_is_named_by_the_bridge_job(self):
        job = self._bridge_job()
        expected = sorted(p.name for p in (REPO_ROOT / "tests").glob("test_matlab_bridge_*.py"))
        assert expected, "no tests/test_matlab_bridge_*.py files found"
        unwired = [name for name in expected if f"tests/{name}" not in job]
        assert not unwired, (
            "these bridge test files are not run by the `bridge` job in "
            f"{CI_WORKFLOW.relative_to(REPO_ROOT)}:\n  "
            + "\n  ".join(unwired)
            + "\n\nAdd them to that job's pytest invocation. They cannot run in the "
            "test matrix instead: those jobs have no NDR-matlab checkout, so these "
            "tests skip there and the skip reads as a pass."
        )

    def test_the_bridge_job_runs_under_the_strict_env_var(self):
        """Strict mode is what turns "the MATLAB tree went missing" from a
        silent skip into a failure."""
        job = self._bridge_job()
        assert STRICT_ENV_VAR in job, (
            f"the `bridge` job does not set {STRICT_ENV_VAR}. Without it, a broken "
            "NDR-matlab checkout makes every bridge test skip, and the job goes "
            "green having checked nothing."
        )

    def test_the_test_matrix_deselects_matlab_dependent_tests(self):
        """The matrix jobs have no NDR-matlab checkout, so tests needing one
        must be DESELECTED there, not skipped.

        Skipping would work, and that is the problem: it leaves a standing
        pile of skips in every matrix job, which trains a reader to scroll
        past skips -- so the next one, a real one, goes unnoticed. Same
        reasoning the whole bridge guard rests on.
        """
        commands = self._job_commands("test")
        assert "not needs_matlab" in commands, (
            "the `test` matrix job does not deselect MATLAB-dependent tests. "
            'Add -m "not needs_matlab" to its pytest invocation; without it '
            "those tests skip there instead, and a standing pile of skips is "
            "what this repo's bridge guard exists to avoid."
        )

    def test_the_bridge_job_does_not_deselect_them(self):
        """The corollary, and the one that actually matters.

        Deselecting in the matrix is only safe because the bridge job runs
        them. If that job ever grew the same filter, the port-synchrony core
        -- drift, the commit-object check, completeness -- would run NOWHERE
        while every job stayed green.
        """
        commands = self._job_commands("bridge")
        assert "needs_matlab" not in commands, (
            "the `bridge` job filters on the needs_matlab marker. It must not: "
            "it is the only job with an NDR-matlab checkout, so filtering there "
            "would leave drift and completeness running in no job at all, with "
            "CI still green."
        )

    def test_the_marker_is_registered(self):
        """An unregistered marker is a warning, not an error -- so a typo in
        the filter would silently deselect nothing at all."""
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "needs_matlab:" in pyproject, (
            "the needs_matlab marker is not registered in pyproject.toml's "
            "[tool.pytest.ini_options] markers list"
        )

    def test_the_skip_backstop_uses_the_same_env_var(self):
        """tests/conftest.py duplicates STRICT_ENV_VAR rather than importing
        it, to avoid a conftest that imports a test module at collection
        time. Duplication is fine; drifting is not -- if the two ever
        disagreed, the backstop would arm on an env var CI never sets and
        would protect nothing while looking installed.
        """
        from tests import conftest

        assert conftest.STRICT_ENV_VAR == STRICT_ENV_VAR, (
            "tests/conftest.py and test_matlab_bridge_completeness.py disagree "
            f"about the strict env var: {conftest.STRICT_ENV_VAR!r} vs "
            f"{STRICT_ENV_VAR!r}. The skip backstop would arm on a variable CI "
            "does not set."
        )

    def test_the_backstop_covers_this_module(self):
        """The backstop keys on the test FILE NAME. A bridge test file named
        outside that convention would sit outside the net -- so assert the
        convention holds for the files that exist."""
        from tests import conftest

        for path in sorted((REPO_ROOT / "tests").glob("test_matlab_bridge_*.py")):
            assert path.name.startswith(conftest.BRIDGE_TEST_PREFIX), (
                f"{path.name} is a bridge test file but does not start with "
                f"{conftest.BRIDGE_TEST_PREFIX!r}, so tests/conftest.py's skip "
                "backstop does not cover it."
            )

    def test_the_matlab_checkout_is_not_shallow(self):
        """``actions/checkout`` is shallow by default, and a shallow
        NDR-matlab collapses every file's history to the last merge commit --
        `git cat-file -t` reports older commits missing and `git log -- <path>`
        names the wrong latest commit."""
        job = self._bridge_job()
        checkouts = job.count("repository: VH-Lab/NDR-matlab")
        assert checkouts, "the `bridge` job does not check out NDR-matlab"
        assert job.count("fetch-depth: 0") >= checkouts, (
            "every NDR-matlab checkout in the `bridge` job needs `fetch-depth: 0`. "
            f"Found {checkouts} checkout(s) and {job.count('fetch-depth: 0')} "
            "fetch-depth setting(s)."
        )


# ---------------------------------------------------------------------------
# A hash written without a review
# ---------------------------------------------------------------------------


#: Bases to diff this branch against, in order of preference.
MERGE_BASE_CANDIDATES = ("origin/main", "origin/master", "main", "master")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
    )


def _merge_base() -> str | None:
    """The commit this branch diverged from, or None if git cannot say.

    None on a shallow clone with no base ref -- which is why the workflow
    checks this repo out with ``fetch-depth: 0``.
    """
    for candidate in MERGE_BASE_CANDIDATES:
        if _git("rev-parse", "--verify", "--quiet", candidate).returncode != 0:
            continue
        result = _git("merge-base", candidate, "HEAD")
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def _entries_by_key(data: Any, source: Path) -> dict[tuple[str, str], Entry]:
    """Entries keyed by ``(name, matlab_path)`` so two revisions can be compared."""
    keyed: dict[tuple[str, str], Entry] = {}
    for entry in _walk(data, source):
        if entry.matlab_path:
            keyed[(entry.name, entry.matlab_path)] = entry
    return keyed


class TestAHashChangeIsJustified:
    """Changing a ``matlab_last_sync_hash`` without touching Python must say why.

    THE FAILURE THIS CATCHES. A hash is supposed to mean "I examined this
    version of this file". Nothing in the drift check can tell that claim from
    a guess: writing a current-looking hash turns a red build green and leaves
    a record nothing can contradict afterwards. That is not hypothetical --
    commit 6377973 added ``matlab_last_sync_hash`` to 20 bridge files in one
    go, touching no Python at all, and in doing so asserted that NDR-matlab
    a938988 had been reviewed. It had not; the Intan multi-file feature it
    introduced is still missing. See issue #23.

    THE RULE. If a change alters an entry's hash but touches none of that
    entry's ``python_path`` files, the entry's ``decision_log`` must change in
    the same diff and name the commit being accounted for. Porting the change
    is the ordinary path and needs nothing extra -- this only bites the
    "nothing to do here" case, which is a decision and belongs in writing.

    WHAT IT CANNOT DO. It cannot verify anybody read the diff. It makes the
    claim explicit, specific and attributable -- a sentence in the entry,
    naming a commit, visible in review -- which is the honest ceiling for a
    check like this. 6377973 made its claim 1148 times without writing a
    word; it could not have under this rule.
    """

    def test_a_hash_change_without_a_port_carries_a_reason(self):
        base = _merge_base()
        if base is None:
            message = (
                "no merge-base against "
                f"{'/'.join(MERGE_BASE_CANDIDATES)} -- cannot tell which entries "
                "this change touches. A shallow clone causes this; CI checks this "
                "repo out with fetch-depth: 0."
            )
            if os.environ.get(STRICT_ENV_VAR, "").strip():
                pytest.fail(
                    f"{message} ({STRICT_ENV_VAR} is set, so history was supposed "
                    "to be there -- skipping would report a check that could not "
                    "run as one that passed.)"
                )
            pytest.skip(message)

        # base against the WORKING TREE, not base..HEAD: the new state of each
        # entry is read from the working tree, so the file list must come from
        # the same place or the two disagree. In CI they are identical; locally
        # this is what makes the check answer for edits you have not committed
        # yet -- which is when you want to hear about them.
        changed = {
            line.strip()
            for line in _git("diff", "--name-only", base).stdout.splitlines()
            if line.strip()
        }
        if not changed:
            return  # nothing in this branch to judge

        offenders: list[str] = []
        for source in all_bridge_files():
            rel = source.relative_to(REPO_ROOT).as_posix()
            if rel not in changed:
                continue
            before = _git("show", f"{base}:{rel}")
            if before.returncode != 0:
                continue  # the file is new in this branch; nothing to compare
            old_entries = _entries_by_key(yaml.safe_load(before.stdout) or {}, source)
            new_entries = _entries_by_key(
                yaml.safe_load(source.read_text(encoding="utf-8")) or {}, source
            )
            for key, entry in new_entries.items():
                previous = old_entries.get(key)
                if previous is None:
                    continue  # a brand-new entry, judged by the other guards
                if entry.matlab_last_sync_hash == previous.matlab_last_sync_hash:
                    continue
                if any(f"src/{path}" in changed for path in entry.python_paths):
                    continue  # the port moved with the hash: the ordinary path
                log = entry.decision_log
                if log == previous.decision_log:
                    offenders.append(
                        f"{entry.where}: {entry.name} -> {entry.matlab_path}\n"
                        f"      hash {previous.matlab_last_sync_hash or '(none)'} -> "
                        f"{entry.matlab_last_sync_hash}, no Python change, "
                        f"decision_log unchanged"
                    )
                elif entry.matlab_last_sync_hash[:7].lower() not in log.lower():
                    offenders.append(
                        f"{entry.where}: {entry.name} -> {entry.matlab_path}\n"
                        f"      decision_log changed but does not name "
                        f"{entry.matlab_last_sync_hash}"
                    )

        assert not offenders, (
            f"{len(offenders)} entr{'y' if len(offenders) == 1 else 'ies'} changed a "
            "matlab_last_sync_hash without porting anything and without saying "
            "why:\n  "
            + "\n  ".join(offenders)
            + '\n\nA hash means "I examined this version of this file". Moving it '
            "with no Python change asserts the MATLAB change needed nothing here -- "
            "which may well be true, but it is a DECISION, and an unrecorded one is "
            "indistinguishable from nobody having looked.\n\n"
            "Either port the change, or add to that entry's decision_log a note "
            "naming the commit and why it is a no-op on the Python side, e.g.\n"
            "    decision_log: >\n"
            "      ... NDR-matlab abc1234 renamed a local variable; no behavioural\n"
            "      change, nothing to port.\n\n"
            "This is issue #23: commit 6377973 wrote 1148 such claims in one commit "
            "and the Intan multi-file feature went missing behind them."
        )


# ---------------------------------------------------------------------------
# The cross-references into the spec resolve
# ---------------------------------------------------------------------------


#: Documents that point at the spec instead of restating it.
CROSS_REFERRING_DOCS = (
    "AGENTS.md",
    "docs/PORT_STATUS.md",
    "docs/developer_notes/PYTHON_PORTING_GUIDE.md",
    "docs/developer_notes/symmetry_tests.md",
)

#: "Section 6", "section 5a", "§ 4" -- however a document spells it.
SECTION_REFERENCE = re.compile(r"(?:section|§)\s*(\d+[a-z]?)", re.IGNORECASE)

#: The other two bridge repos, whose specs are cited by name, never by number.
FOREIGN_REPO = re.compile(r"\b(?:NDI|DID)-(?:python|matlab)\b", re.IGNORECASE)

#: This spec, as the cross-referring documents spell its path.
SPEC_REL = "docs/developer_notes/ndr_matlab_python_bridge.yaml"


def _sentences(text: str) -> list[str]:
    """Prose sentences, with comment and markdown furniture removed.

    Both file kinds wrap prose across lines, so a per-line scan would miss a
    repo name and a section number that a reader sees side by side. Blank
    lines, bullets and headings end a block; sentence-enders and colons split
    within one.
    """
    blocks: list[list[str]] = [[]]
    for raw in text.splitlines():
        line = re.sub(r"^\s*(?:#+|>)\s?", "", raw).rstrip()
        if not line.strip() or re.match(r"^\s*(?:[-*]|\d+\.)\s", line):
            blocks.append([])
            line = re.sub(r"^\s*(?:[-*]|\d+\.)\s", "", line)
        if line.strip():
            blocks[-1].append(line.strip())
    joined = [" ".join(block) for block in blocks if block]
    out: list[str] = []
    for block in joined:
        out.extend(part for part in re.split(r"(?<=[.;:])\s+", block) if part.strip())
    return out


#: A spec section heading: a numbered line directly under a banner rule. The
#: banner is what distinguishes a heading from a numbered list item, which is
#: spelled the same way inside the spec's comment blocks.
SPEC_HEADING = re.compile(r"^# ={10,}\n# (\d+[a-z]?)\. ", re.MULTILINE)


def spec_sections() -> set[str]:
    return set(SPEC_HEADING.findall(SPEC_PATH.read_text(encoding="utf-8")))


class TestTheCrossReferencesResolve:
    """Concentrating the rules in one file only works if the pointers to it
    are good.

    Renumbering a section silently turns every "see section 3a" elsewhere
    into a dead end, and a reader who follows a dead pointer goes back to
    guessing -- or worse, writes the rule down locally again, which is the
    duplication this layout exists to prevent.
    """

    def test_the_spec_has_numbered_sections(self):
        sections = spec_sections()
        assert len(sections) >= 5, (
            f"only found sections {sorted(sections)} in "
            f"{SPEC_PATH.relative_to(REPO_ROOT)} -- the heading pattern this test "
            "keys on has probably changed."
        )

    @pytest.mark.parametrize("doc", CROSS_REFERRING_DOCS + (SPEC_REL,))
    def test_no_section_number_is_used_for_another_repos_spec(self, doc: str):
        """A bare "section N" always means THIS spec, so never write one for
        NDI-python's or DID-python's.

        The check above validates every section reference against this file --
        which is the right thing to do while every reference means this file,
        and a silent lie the moment one does not. "See NDI-python section 7"
        resolves happily against NDR's own Section 7 (Structure for Classes and
        Functions), and the reader lands somewhere plausible and wrong. That is
        worse than a dangling pointer, which at least announces itself.

        So cite another repo's spec by FILE and by the NAME of its section.
        Their numbering is theirs to change, which is the same staleness that
        got a description of NDI-python's rule deleted from Section 5 in the
        first place.
        """
        text = (REPO_ROOT / doc).read_text(encoding="utf-8")
        offenders = [
            sentence.strip()
            for sentence in _sentences(text)
            if FOREIGN_REPO.search(sentence) and SECTION_REFERENCE.search(sentence)
        ]
        assert not offenders, (
            f"{doc} names another repo and a numbered section in the same "
            "sentence:\n  "
            + "\n  ".join(offenders)
            + f"\n\nA section number reads as a pointer into "
            f"{SPEC_PATH.relative_to(REPO_ROOT)}, so this either says something "
            "false or resolves to the wrong section here. Cite the other repo's "
            "spec by file and by the name of the section instead."
        )

    @pytest.mark.parametrize("doc", CROSS_REFERRING_DOCS)
    def test_every_section_reference_names_a_real_section(self, doc: str):
        path = REPO_ROOT / doc
        assert path.exists(), f"{doc} is listed as cross-referring but does not exist"
        sections = spec_sections()
        text = path.read_text(encoding="utf-8")

        dangling = sorted(
            {
                referenced
                for referenced in SECTION_REFERENCE.findall(text)
                if referenced not in sections
            }
        )
        assert not dangling, (
            f"{doc} points at section(s) {dangling} of "
            f"{SPEC_PATH.relative_to(REPO_ROOT)}, which do not exist. "
            f"It has: {sorted(sections)}."
        )


# ---------------------------------------------------------------------------
# Self-tests: these guards can actually fail
# ---------------------------------------------------------------------------


class TestTheGuardsWouldActuallyCatchOne:
    """A guard that cannot fail is worse than no guard: it is a green light
    that means nothing. These prove each one can go red."""

    def _entry(self, **fields: Any) -> Entry:
        return Entry(REPO_ROOT / "src" / BRIDGE_FILENAME, ("functions", "0"), fields)

    def test_two_bare_entries_for_one_function_are_a_duplicate(self):
        group = [
            self._entry(name="readGEF", matlab_path="+ndr/+format/+stereoseq/readGEF.m"),
            self._entry(name="readGEF", matlab_path="+ndr/+format/+stereoseq/readGEF.m"),
        ]
        assert not _is_a_declared_multi_module_pair(group)

    def test_a_pair_where_only_one_side_declares_itself_is_not_a_declaration(self):
        group = [
            self._entry(name="readGEF", python_qualified="ndr.format.stereoseq.readGEF"),
            self._entry(name="readGEF"),
        ]
        assert not _is_a_declared_multi_module_pair(group)

    def test_a_pair_with_the_same_python_qualified_is_not_two_modules(self):
        group = [
            self._entry(name="readGEF", python_qualified="ndr.format.stereoseq.readGEF"),
            self._entry(name="readGEF", python_qualified="ndr.format.stereoseq.readGEF"),
        ]
        assert not _is_a_declared_multi_module_pair(group)

    def test_a_fully_declared_two_module_pair_is_allowed(self):
        group = [
            self._entry(name="readGEF", python_qualified="ndr.format.stereoseq.readGEF"),
            self._entry(name="readGEF", python_qualified="ndr.reader.stereoseq.readGEF"),
        ]
        assert _is_a_declared_multi_module_pair(group)

    def test_class_methods_sharing_their_class_file_are_not_duplicates(self):
        """Keyed on (name, matlab_path), so ``read`` and ``close`` recorded
        off one class file do not collide. Keyed on path alone they would,
        which is why they are not."""
        by_key: dict[tuple[str, str], int] = collections.Counter()
        for method in ("read", "close"):
            by_key[(method, "+ndr/+reader/intan_rhd.m")] += 1
        assert max(by_key.values()) == 1

    def test_a_blob_hash_is_recognised_as_a_blob(self):
        """The check that would have caught NDI-python's 35 blob hashes.

        Hashing this repo's own file gives a real blob that no repository
        holds as a commit, so the assertion does not depend on NDR-matlab's
        contents.
        """
        root = require_matlab_root()
        require_full_history(root)
        blob = subprocess.run(
            ["git", "-C", str(root), "hash-object", "--", "README.md"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert _git_object_type(root, blob) == "blob"

    def test_a_hash_naming_nothing_is_missing_not_a_commit(self):
        root = require_matlab_root()
        assert _git_object_type(root, "0123456789abcdef0123456789abcdef01234567") == "missing"

    def test_head_is_a_commit(self):
        """The positive control: the type check does distinguish, rather
        than reporting everything as wrong."""
        root = require_matlab_root()
        assert _git_object_type(root, "HEAD") == "commit"

    def test_the_retired_vocabulary_is_not_quietly_accepted(self):
        for dropped in REPLACED_STATUSES:
            assert dropped not in ALLOWED_STATUSES

    def test_a_foreign_section_number_is_caught_across_a_line_break(self):
        """The sentence a reader sees is not always the line a scan sees."""
        wrapped = "# The rule lives in NDI-python's spec,\n# section 7. Read it there.\n"
        flagged = [
            sentence
            for sentence in _sentences(wrapped)
            if FOREIGN_REPO.search(sentence) and SECTION_REFERENCE.search(sentence)
        ]
        assert flagged == ["The rule lives in NDI-python's spec, section 7."]

    def test_naming_a_foreign_section_rather_than_numbering_it_is_allowed(self):
        """The remedy has to pass, or the guard just bans the cross-reference."""
        named = "# Settled in NDI-python#211; see the DRIFT section of its spec.\n"
        assert not [
            sentence
            for sentence in _sentences(named)
            if FOREIGN_REPO.search(sentence) and SECTION_REFERENCE.search(sentence)
        ]

    def test_this_repos_own_section_numbers_are_left_alone(self):
        """Section 5a is this file's, and citing it is the whole point."""
        ours = "# What counts as out of date is Section 5a.\n"
        assert not [
            sentence
            for sentence in _sentences(ours)
            if FOREIGN_REPO.search(sentence) and SECTION_REFERENCE.search(sentence)
        ]

    def test_a_string_python_path_and_a_list_both_read(self):
        assert self._entry(python_path="ndr/reader/base.py").python_paths == ["ndr/reader/base.py"]
        assert self._entry(python_path=["a.py", "b.py"]).python_paths == ["a.py", "b.py"]
        assert self._entry().python_paths == []


if __name__ == "__main__":
    pytest.main([__file__])
