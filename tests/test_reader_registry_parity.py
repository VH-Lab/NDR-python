"""Every reader alias NDR-matlab accepts must resolve here too.

The registry is a name-to-class map, and a name that works in one language
but not the other is a portability trap: code written against NDR-matlab
fails at the ndr_reader() call with "do not know how to make a reader".

Expected aliases come from NDR-matlab resource/ndr_reader_types.json. The
MATLAB tree is not on the path here, so they are listed explicitly; the
symmetry job is what keeps the two files honest against each other.
"""

import pytest

from ndr.fun.ndrresource import ndrresource

# NDR-matlab resource/ndr_reader_types.json, alias -> reader module.
MATLAB_ALIASES = {
    "intan": "ndr.reader.intan_rhd",
    "RHD": "ndr.reader.intan_rhd",
    "intanRHD": "ndr.reader.intan_rhd",
    "ced-smr": "ndr.reader.ced_smr",
    "smr": "ndr.reader.ced_smr",
    "son": "ndr.reader.ced_smr",
    "smrx": "ndr.reader.ced_smr",
    "ced-smrx": "ndr.reader.ced_smr",
    "SpikeGadgets": "ndr.reader.spikegadgets_rec",
    "SpikeGadgetsREC": "ndr.reader.spikegadgets_rec",
    "rec": "ndr.reader.spikegadgets_rec",
    "neo": "ndr.reader.neo",
    "sev": "ndr.reader.tdt_sev",
    "tdt_sev": "ndr.reader.tdt_sev",
    "abf": "ndr.reader.axon_abf",
    "axon_abf": "ndr.reader.axon_abf",
    "whitematter": "ndr.reader.whitematter",
    "WMHS": "ndr.reader.whitematter",
    "bjg": "ndr.reader.bjg",
    "bjg_bin": "ndr.reader.bjg",
    "dabrowska": "ndr.reader.dabrowska",
    "dabrowska_mat": "ndr.reader.dabrowska",
    "neuropixelsGLX": "ndr.reader.neuropixelsGLX",
    "neuropixels": "ndr.reader.neuropixelsGLX",
    "spikeglx": "ndr.reader.neuropixelsGLX",
    "imec": "ndr.reader.neuropixelsGLX",
    "tiffstack": "ndr.reader.tiffstack",
    "tiff": "ndr.reader.tiffstack",
    "tif": "ndr.reader.tiffstack",
    "multipagetiff": "ndr.reader.tiffstack",
    "prairieview": "ndr.reader.prairieview",
    "prairie": "ndr.reader.prairieview",
    "pv": "ndr.reader.prairieview",
    "prairieview_pcf": "ndr.reader.prairieview",
    "vld": "ndr.reader.vld",
    "vlh": "ndr.reader.vld",
    "vhlv": "ndr.reader.vld",
    "vhlabview": "ndr.reader.vld",
    "labview": "ndr.reader.vld",
}

# ndr.reader.imagestack exists in NDR-matlab with aliases imagestack / nansen /
# nansenimagestack. There is no Python port of it yet, so registering the name
# would resolve to a class that does not exist. Tracked as a porting gap.
KNOWN_UNPORTED = {"imagestack", "nansen", "nansenimagestack"}


def _registry():
    return ndrresource("ndr_reader_types.json")


def _lookup(name):
    for entry in _registry():
        if name.lower() in [t.lower() for t in entry["type"]]:
            return entry
    return None


@pytest.mark.parametrize("alias,module", sorted(MATLAB_ALIASES.items()))
def test_matlab_alias_resolves(alias, module):
    entry = _lookup(alias)
    assert entry is not None, f"alias '{alias}' works in NDR-matlab but not here"
    assert entry["classname"].startswith(module + "."), (
        f"alias '{alias}' resolves to {entry['classname']}, expected a {module} reader"
    )


def test_unported_readers_are_not_registered():
    """Registering a name whose class does not exist trades one error for a worse one."""
    for alias in KNOWN_UNPORTED:
        assert _lookup(alias) is None, (
            f"'{alias}' is registered but ndr.reader.imagestack has no Python port"
        )


def test_every_registered_class_is_importable():
    """A registry entry that cannot be imported fails only at read time."""
    import importlib

    for entry in _registry():
        module_path, class_name = entry["classname"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        assert hasattr(module, class_name), f"{entry['classname']} is not importable"


def test_no_alias_is_claimed_by_two_readers():
    seen = {}
    for entry in _registry():
        for t in entry["type"]:
            key = t.lower()
            assert key not in seen, f"alias '{t}' maps to both {seen[key]} and {entry['classname']}"
            seen[key] = entry["classname"]
