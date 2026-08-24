"""Automated tests for the (legacy) Prairie View reader.

Ported from tools/tests/+ndr/+unittest/+reader/TestPrairieView.m

ndr.reader.prairieview reads a legacy Prairie recording: a directory of
one-TIFF-per-frame plus a '*_Main.pcf' config whose '[Image TimeStamp (us)]'
section holds per-frame timestamps (or a Prairie '.xml' with per-frame times).
These tests synthesize such recordings (single-plane TIFFs + hand-written
config files) into a temporary directory and check geometry, frame round-trip,
multi-channel channel-on-C grouping, multi-cycle spanning, and that timestamps
come from the config in seconds. No external example data is required.

The ground-truth image content mirrors the MATLAB test's
``reshape(1:(Y*X), Y, X)`` construction. MATLAB reshape is column-major, so the
Python truth is built with ``order='F'`` and written verbatim as the TIFF pixel
data (tifffile round-trips it), guaranteeing a byte-exact comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from ndr.format import prairieview as pv_format
from ndr.reader.prairieview import ndr_reader_prairieview

Y = 9  # image height  (Lines per frame)
X = 7  # image width   (Pixels per line)
T = 5  # number of frames / images


# ----------------------------------------------------------------------
# Fixture writers (Python ports of the MATLAB static writers)
# ----------------------------------------------------------------------


def _base_plane(i0: int) -> np.ndarray:
    """reshape(1:(Y*X), Y, X) column-major, plus offset i0, as uint16."""
    plane = np.reshape(np.arange(1, Y * X + 1), (Y, X), order="F")
    return (plane + i0).astype(np.uint16)


def _write_tiff(path, img: np.ndarray) -> None:
    tifffile.imwrite(str(path), np.asarray(img, dtype=np.uint16))


def _write_pcf(path, y: int, x: int, t: int, times_us) -> None:
    """Minimal legacy Prairie '.pcf' ([Main] then [Image TimeStamp (us)])."""
    lines = []
    lines.append("[Main]")
    lines.append(f"Total images = {t}")
    lines.append(f"Lines per frame = {y}")
    lines.append(f"Pixels per line = {x}")
    lines.append(f"Frame period (us) = {250000}")
    lines.append("")
    lines.append("[Image TimeStamp (us)]")
    for i in range(t):
        lines.append(f"{i + 1}=%.15g" % times_us[i])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_multicycle_pcf(path, y: int, x: int, cycle_counts, times_us) -> None:
    """Real-style multi-cycle legacy .pcf spanning every cycle's frames."""
    total = int(np.sum(cycle_counts))
    lines = []
    lines.append("[Main]")
    lines.append("Acquisition type=TSERIES_MAIN")
    lines.append("Bit depth=12")
    lines.append("Channel 1 active=True")
    lines.append("Frame period (us)=1486848.0")
    lines.append(f"Lines per frame={y}")
    lines.append(f"Pixels per line={x}")
    lines.append(f"Total cycles={len(cycle_counts)}")
    lines.append(f"Total images={total}")
    lines.append("Version=2.1.0.2")
    lines.append("")
    for k in range(len(cycle_counts)):
        lines.append(f"[Cycle {k + 1}]")
        lines.append("Acquisition type=TSERIES_CYCLE")
        lines.append("Number of frames to average=1")
        lines.append(f"Number of images={cycle_counts[k]}")
        lines.append("Period (us)=0.0")
        lines.append("")
    lines.append("[Image TimeStamp (us)]")
    for i in range(total):
        lines.append(f"{i + 1}=%.15g" % times_us[i])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_pvscan_xml(path, y: int, x: int, times_sec) -> None:
    """Real Prairie v4 PVScan layout: one <Sequence cycle> per timepoint."""
    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append('<PVScan version="4.0.0.43" date="9/28/2018 5:40:34 PM" notes="">')
    for i in range(len(times_sec)):
        lines.append(f'  <Sequence type="TSeries Timed Element" cycle="{i + 1}">')
        lines.append(
            '    <Frame relativeTime="0" absoluteTime="%.12g" index="1" '
            'label="CurrentSettings">' % times_sec[i]
        )
        lines.append(
            '      <File channel="1" channelName="Ch1" '
            'filename="t00004-001_Cycle%03d_CurrentSettings_Ch1_000001.tif" />' % (i + 1)
        )
        lines.append(
            '      <File channel="2" channelName="Ch2" '
            'filename="t00004-001_Cycle%03d_CurrentSettings_Ch2_000001.tif" />' % (i + 1)
        )
        lines.append("      <PVStateShard>")
        lines.append(
            '        <Key key="linesPerFrame" permissions="Read, Write, Save" ' 'value="%d" />' % y
        )
        lines.append(
            '        <Key key="pixelsPerLine" permissions="Read, Write, Save" ' 'value="%d" />' % x
        )
        lines.append(
            '        <Key key="framePeriod" permissions="Read, Write, Save" ' 'value="1.4819328" />'
        )
        lines.append(
            '        <Key key="dwellTime" permissions="Read, Write, Save" ' 'value="3.6" />'
        )
        lines.append("      </PVStateShard>")
        lines.append("    </Frame>")
        lines.append("  </Sequence>")
    lines.append("</PVScan>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_v2_xml(path, y: int, x: int, times_ms) -> None:
    """Legacy v2.2 '.NET DataSet' XML with an embedded <xs:schema> to skip."""
    lines = []
    lines.append('<?xml version="1.0" standalone="yes"?>')
    lines.append("<Acquisition>")
    lines.append('  <xs:schema id="Acquisition" ' 'xmlns:xs="http://www.w3.org/2001/XMLSchema">')
    lines.append('    <xs:element name="Lines_Per_Frame" type="xs:double" minOccurs="0" />')
    lines.append('    <xs:element name="Pixels_Per_Line" type="xs:double" minOccurs="0" />')
    lines.append('    <xs:element name="Framerate" type="xs:double" minOccurs="0" />')
    lines.append('    <xs:element name="Time" type="xs:double" minOccurs="0" />')
    lines.append("  </xs:schema>")
    lines.append("  <Acquisition_Header>")
    lines.append(f"    <Lines_Per_Frame>{y}</Lines_Per_Frame>")
    lines.append(f"    <Pixels_Per_Line>{x}</Pixels_Per_Line>")
    lines.append("    <Framerate>0.9</Framerate>")
    lines.append(f"    <Total_Frames>{len(times_ms)}</Total_Frames>")
    lines.append("  </Acquisition_Header>")
    for i in range(len(times_ms)):
        lines.append("  <Dataset_x0020_2>")
        lines.append(
            "    <Channel_1_Filename>t00001-001_Cycle%03d_Ch1_000001.tif"
            "</Channel_1_Filename>" % (i + 1)
        )
        lines.append(
            "    <Channel_2_Filename>t00001-001_Cycle%03d_Ch2_000001.tif"
            "</Channel_2_Filename>" % (i + 1)
        )
        lines.append("    <Frame>1</Frame>")
        lines.append("    <Time>%.15g</Time>" % times_ms[i])
        lines.append("  </Dataset_x0020_2>")
    lines.append("</Acquisition>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# Recording fixtures
# ----------------------------------------------------------------------


TIMES_US = [0, 90000, 250000, 260000, 600000]
MULTI_TIMES_US = [0, 100000, 250000, 270000, 500000]
MULTI_C = 2
XML_TIMES_SEC = [0.329333, 2.132962, 3.976685]
XML_C = 2
V2_TIMES_MS = [0, 1468.75, 2875]
V2_C = 2
CYC_COUNTS = [1, 4, 1]


@pytest.fixture(scope="module")
def reader() -> ndr_reader_prairieview:
    return ndr_reader_prairieview()


@pytest.fixture(scope="module")
def single_pcf(tmp_path_factory):
    """Single-channel legacy .pcf recording; returns (dir, config, truth)."""
    d = tmp_path_factory.mktemp("Recording-001")
    truth = np.zeros((Y, X, 1, 1, T), dtype=np.uint16)
    for i in range(T):
        truth[:, :, 0, 0, i] = _base_plane(i * 100)
        fn = d / ("Recording_Cycle001_Ch2_%06d.tif" % (i + 1))
        _write_tiff(fn, truth[:, :, 0, 0, i])
    cfg = d / "Recording_Main.pcf"
    _write_pcf(cfg, Y, X, T, TIMES_US)
    return {"dir": str(d), "config": str(cfg), "truth": truth}


@pytest.fixture(scope="module")
def multi_pcf(tmp_path_factory):
    """Two-channel legacy .pcf recording; channels on the C axis."""
    d = tmp_path_factory.mktemp("Recording-002")
    mtruth = np.zeros((Y, X, MULTI_C, 1, T), dtype=np.uint16)
    for c in range(MULTI_C):  # c = 0,1 -> channel 1,2
        for i in range(T):
            mtruth[:, :, c, 0, i] = _base_plane(i * 100 + (c + 1) * 10000)
            fn = d / ("Rec_Cycle001_Ch%d_%06d.tif" % (c + 1, i + 1))
            _write_tiff(fn, mtruth[:, :, c, 0, i])
    _write_pcf(d / "Rec_Main.pcf", Y, X, T, MULTI_TIMES_US)
    return {"dir": str(d), "truth": mtruth}


@pytest.fixture(scope="module")
def xml_pvscan(tmp_path_factory):
    """Modern PVScan XML two-channel recording (per-frame absoluteTime)."""
    d = tmp_path_factory.mktemp("Recording-XML")
    txml = len(XML_TIMES_SEC)
    xtruth = np.zeros((Y, X, XML_C, 1, txml), dtype=np.uint16)
    for c in range(XML_C):
        for i in range(txml):
            xtruth[:, :, c, 0, i] = _base_plane(i * 100 + (c + 1) * 5000)
            fn = d / ("t00004-001_Cycle%03d_CurrentSettings_Ch%d_000001.tif" % (i + 1, c + 1))
            _write_tiff(fn, xtruth[:, :, c, 0, i])
    _write_pvscan_xml(d / "t00004-001.xml", Y, X, XML_TIMES_SEC)
    return {"dir": str(d), "truth": xtruth}


@pytest.fixture(scope="module")
def xml_v2(tmp_path_factory):
    """Legacy v2.2 '.NET DataSet' XML two-channel recording (<Time> ms)."""
    d = tmp_path_factory.mktemp("Recording-v2")
    tv2 = len(V2_TIMES_MS)
    v2truth = np.zeros((Y, X, V2_C, 1, tv2), dtype=np.uint16)
    for c in range(V2_C):
        for i in range(tv2):
            v2truth[:, :, c, 0, i] = _base_plane(i * 100 + (c + 1) * 3000)
            fn = d / ("t00001-001_Cycle%03d_Ch%d_000001.tif" % (i + 1, c + 1))
            _write_tiff(fn, v2truth[:, :, c, 0, i])
    _write_v2_xml(d / "t00001-001.xml", Y, X, V2_TIMES_MS)
    return {"dir": str(d), "truth": v2truth}


@pytest.fixture(scope="module")
def multicycle_pcf(tmp_path_factory):
    """Multi-cycle .pcf: one epoch spans all cycles (cycle-then-frame order)."""
    d = tmp_path_factory.mktemp("t00012-001")
    tcyc = int(np.sum(CYC_COUNTS))
    cyctruth = np.zeros((Y, X, 1, 1, tcyc), dtype=np.uint16)
    tp = 0
    for cyc in range(len(CYC_COUNTS)):
        for fr in range(CYC_COUNTS[cyc]):
            cyctruth[:, :, 0, 0, tp] = _base_plane(tp * 100)
            fn = d / ("t00012-001_Cycle%03d_CurrentSettings_Ch1_%06d.tif" % (cyc + 1, fr + 1))
            _write_tiff(fn, cyctruth[:, :, 0, 0, tp])
            tp += 1
    times_us = np.arange(tcyc, dtype=float) * 1486848.0
    _write_multicycle_pcf(d / "t00012-001_Main.pcf", Y, X, CYC_COUNTS, times_us)
    return {"dir": str(d), "truth": cyctruth, "times_us": times_us}


# ----------------------------------------------------------------------
# Tests: legacy .pcf single channel
# ----------------------------------------------------------------------


def test_config_parsing(single_pcf):
    v = pv_format.readconfig(single_pcf["dir"])
    assert v["is_xml"] is False
    assert v["Main"]["Total_images"] == T
    assert v["Main"]["Lines_per_frame"] == Y
    assert v["Main"]["Pixels_per_line"] == X
    np.testing.assert_array_equal(
        np.asarray(v["Image_TimeStamp__us_"]).ravel(), np.asarray(TIMES_US, float)
    )


def test_config_filename_discovery(single_pcf):
    from_dir = pv_format.configfilename(single_pcf["dir"])
    assert from_dir == single_pcf["config"]


def test_geometry(reader, single_pcf):
    ef = [single_pcf["dir"]]
    assert reader.numframes(ef, 1) == T
    assert reader.framesize(ef, 1) == [Y, X, 1, 1, T]
    assert reader.datatype(ef, 1) == "uint16"


def test_frames_round_trip(reader, single_pcf):
    ef = [single_pcf["dir"]]
    frames = reader.readframes(ef, 1)
    np.testing.assert_array_equal(frames, single_pcf["truth"])


def test_timestamps_from_config(reader, single_pcf):
    ef = [single_pcf["dir"]]
    ec = reader.epochclock(ef, 1)
    assert ec[0].type == "dev_local_time"
    ft = reader.frametimes(ef, 1)
    np.testing.assert_allclose(ft.ravel(), np.asarray(TIMES_US, float) / 1e6, atol=1e-12)
    t0t1 = reader.t0_t1(ef, 1)
    np.testing.assert_allclose(t0t1[0], [TIMES_US[0] / 1e6, TIMES_US[-1] / 1e6], atol=1e-12)
    # subset request (1-based [2, 4] -> values at those frame indices)
    ftsub = reader.frametimes(ef, 1, [2, 4])
    np.testing.assert_allclose(
        ftsub.ravel(),
        np.asarray([TIMES_US[1], TIMES_US[3]], float) / 1e6,
        atol=1e-12,
    )


def test_anchor_on_config_file(reader, single_pcf):
    by_cfg = reader.readframes([single_pcf["config"]], 1)
    by_dir = reader.readframes([single_pcf["dir"]], 1)
    np.testing.assert_array_equal(by_cfg, by_dir)
    ft_cfg = reader.frametimes([single_pcf["config"]], 1)
    np.testing.assert_allclose(ft_cfg.ravel(), np.asarray(TIMES_US, float) / 1e6, atol=1e-12)


# ----------------------------------------------------------------------
# Tests: multi-channel .pcf
# ----------------------------------------------------------------------


def test_multichannel_geometry(reader, multi_pcf):
    ef = [multi_pcf["dir"]]
    assert reader.numframes(ef, 1) == T
    assert reader.framesize(ef, 1) == [Y, X, MULTI_C, 1, T]


def test_multichannel_frames_round_trip(reader, multi_pcf):
    ef = [multi_pcf["dir"]]
    frames = reader.readframes(ef, 1)
    np.testing.assert_array_equal(frames, multi_pcf["truth"])
    # a single timepoint carries both channels (1-based index 3)
    one = reader.readframes(ef, 1, 3)
    np.testing.assert_array_equal(one, multi_pcf["truth"][:, :, :, :, 2:3])


def test_multichannel_times_per_timepoint(reader, multi_pcf):
    ef = [multi_pcf["dir"]]
    ft = reader.frametimes(ef, 1)
    assert ft.size == T
    np.testing.assert_allclose(ft.ravel(), np.asarray(MULTI_TIMES_US, float) / 1e6, atol=1e-12)
    ec = reader.epochclock(ef, 1)
    assert ec[0].type == "dev_local_time"


# ----------------------------------------------------------------------
# Tests: modern PVScan XML
# ----------------------------------------------------------------------


def test_xml_config_parsing(xml_pvscan):
    v = pv_format.readconfig(xml_pvscan["dir"])
    assert v["is_xml"] is True
    assert v["Main"]["Lines_per_frame"] == Y
    assert v["Main"]["Pixels_per_line"] == X
    np.testing.assert_allclose(
        np.asarray(v["Image_TimeStamp__us_"]).ravel(),
        np.asarray(XML_TIMES_SEC, float) * 1e6,
        atol=1e-3,
    )


def test_xml_geometry_and_frames(reader, xml_pvscan):
    ef = [xml_pvscan["dir"]]
    assert reader.numframes(ef, 1) == len(XML_TIMES_SEC)
    assert reader.framesize(ef, 1) == [Y, X, XML_C, 1, len(XML_TIMES_SEC)]
    frames = reader.readframes(ef, 1)
    np.testing.assert_array_equal(frames, xml_pvscan["truth"])


def test_xml_timestamps(reader, xml_pvscan):
    ef = [xml_pvscan["dir"]]
    ec = reader.epochclock(ef, 1)
    assert ec[0].type == "dev_local_time"
    ft = reader.frametimes(ef, 1)
    np.testing.assert_allclose(ft.ravel(), np.asarray(XML_TIMES_SEC, float), atol=1e-9)


# ----------------------------------------------------------------------
# Tests: legacy v2.2 '.NET DataSet' XML
# ----------------------------------------------------------------------


def test_v2_config_parsing(xml_v2):
    # the embedded XSD schema must be skipped: dims come from the data
    v = pv_format.readconfig(xml_v2["dir"])
    assert v["is_xml"] is True
    assert v["Main"]["Lines_per_frame"] == Y
    assert v["Main"]["Pixels_per_line"] == X
    np.testing.assert_allclose(
        np.asarray(v["Image_TimeStamp__us_"]).ravel(),
        np.asarray(V2_TIMES_MS, float) * 1e3,
        atol=1e-6,
    )


def test_v2_geometry_and_times(reader, xml_v2):
    ef = [xml_v2["dir"]]
    assert reader.numframes(ef, 1) == len(V2_TIMES_MS)
    assert reader.framesize(ef, 1) == [Y, X, V2_C, 1, len(V2_TIMES_MS)]
    frames = reader.readframes(ef, 1)
    np.testing.assert_array_equal(frames, xml_v2["truth"])
    ec = reader.epochclock(ef, 1)
    assert ec[0].type == "dev_local_time"
    ft = reader.frametimes(ef, 1)
    np.testing.assert_allclose(ft.ravel(), np.asarray(V2_TIMES_MS, float) / 1e3, atol=1e-9)


# ----------------------------------------------------------------------
# Tests: multi-cycle .pcf (one epoch spans several cycles)
# ----------------------------------------------------------------------


def test_multicycle_pcf_config(multicycle_pcf):
    v = pv_format.readconfig(multicycle_pcf["dir"])
    total = int(np.sum(CYC_COUNTS))
    assert v["Main"]["Total_images"] == total
    assert np.asarray(v["Image_TimeStamp__us_"]).size == total
    np.testing.assert_allclose(
        np.asarray(v["Image_TimeStamp__us_"]).ravel(),
        multicycle_pcf["times_us"],
        atol=1e-6,
    )


def test_multicycle_epoch_spans_cycles(reader, multicycle_pcf):
    ef = [multicycle_pcf["dir"]]
    total = int(np.sum(CYC_COUNTS))
    assert reader.numframes(ef, 1) == total
    assert reader.framesize(ef, 1) == [Y, X, 1, 1, total]
    frames = reader.readframes(ef, 1)
    np.testing.assert_array_equal(frames, multicycle_pcf["truth"])
    ft = reader.frametimes(ef, 1)
    np.testing.assert_allclose(ft.ravel(), multicycle_pcf["times_us"] / 1e6, atol=1e-9)
