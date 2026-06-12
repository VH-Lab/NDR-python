"""Reader-dispatch tests: stub readers fail fast at construction, and the
reconciled ndr_reader_types alias registry resolves both languages' aliases
case-insensitively (audit 6.2-5 stub marking + 6.2-6 single-source/aliases)."""

import pytest

from ndr.known_readers import known_readers
from ndr.reader_wrapper import ndr_reader

# Readers the audit lists as not-yet-implemented; constructing one must fail
# fast with NotImplementedError rather than raising deep inside a read.
STUB_TYPES = ["neo", "spikegadgets_rec", "whitematter", "bjg", "tdt_sev", "dabrowska"]

# Implemented readers: the dispatch should not raise NotImplementedError for
# these (they may need data files to do real work, but must construct).
IMPLEMENTED_TYPES = ["intan_rhd", "axon_abf", "ced_smr", "neuropixelsGLX"]


@pytest.mark.parametrize("rtype", STUB_TYPES)
def test_stub_reader_fails_fast(rtype):
    with pytest.raises(NotImplementedError):
        ndr_reader(rtype)


@pytest.mark.parametrize("rtype", IMPLEMENTED_TYPES)
def test_implemented_reader_not_flagged_stub(rtype):
    # Should not raise NotImplementedError (it may raise nothing, or something
    # else if construction needs resources — we only assert it is not stubbed).
    try:
        ndr_reader(rtype)
    except NotImplementedError:
        pytest.fail(f"{rtype} should not be flagged as an unimplemented stub")
    except Exception:
        pass


def test_matlab_aliases_resolve():
    # MATLAB-only aliases now resolve in Python (case-insensitive). 'RHD' is an
    # implemented reader; 'WMHS' maps to the whitematter stub (so fails fast).
    ndr_reader("RHD")  # intan_rhd, implemented -> constructs
    ndr_reader("intanRHD")
    with pytest.raises(NotImplementedError):
        ndr_reader("WMHS")  # whitematter stub
    with pytest.raises(NotImplementedError):
        ndr_reader("dabrowska_mat")  # dabrowska stub


def test_unknown_type_raises_value_error():
    with pytest.raises(ValueError):
        ndr_reader("definitely_not_a_real_reader")


def test_known_readers_includes_all_entries():
    types = known_readers()
    # Every reader's alias list is present; the registry is well-formed.
    flat = [t for entry in types for t in entry]
    for expected in ["intan_rhd", "neo", "spikegadgets_rec", "neuropixelsGLX"]:
        assert expected in flat
