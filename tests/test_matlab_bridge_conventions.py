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

    def _bridge_job(self) -> str:
        text = CI_WORKFLOW.read_text(encoding="utf-8")
        marker = "\n  bridge:\n"
        assert marker in text, f"{CI_WORKFLOW.relative_to(REPO_ROOT)} has no `bridge:` job"
        after = text.split(marker, 1)[1]
        # Up to the next top-level job key (two-space indent at line start).
        following = re.search(r"\n  [a-zA-Z0-9_-]+:\n", after)
        return after[: following.start()] if following else after

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

    def test_a_string_python_path_and_a_list_both_read(self):
        assert self._entry(python_path="ndr/reader/base.py").python_paths == ["ndr/reader/base.py"]
        assert self._entry(python_path=["a.py", "b.py"]).python_paths == ["a.py", "b.py"]
        assert self._entry().python_paths == []


if __name__ == "__main__":
    pytest.main([__file__])
