"""Intan RHD multi-file recordings read as one continuous stream.

MATLAB counterpart: ``tools/tests/+ndr/+unittest/+reader/TestIntanRhd.m``,
specifically ``testDetectFileMode`` and ``testMultiFileMode``, which arrived
with NDR-matlab 034f210 and a938988.

Intan's acquisition software starts a new ``.rhd`` file each time its
file-length threshold is reached, naming them
``<prefix>_<YYMMDD>_<HHMMSS>.rhd``. The set is one recording. MATLAB gained
that support in 034f210 and made auto-detection the default in a938988;
Python had neither until now, and read only the file it was handed while
reporting that file's duration as the recording's -- silently, with no error.

FIXTURES. The same strategy MATLAB's own tests use: copy
``example_data/example.rhd`` under timestamped names. Two copies of one file
are not a physically meaningful recording, but they are exactly right for
what is under test -- the file-set discovery, the block arithmetic and the
range mapping -- and they make the expected answers arithmetic rather than
approximate: two copies must read as exactly twice one copy, with each half
equal to it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from ndr.format.intan import (
    Intan_RHD2000_blockinfo,
    detectRHD2000FileMode,
    getRHD2000FileList,
    read_Intan_RHD2000_datafile,
    read_Intan_RHD2000_header,
)
from ndr.fun.ndrpath import ndrpath
from ndr.reader.intan_rhd import ndr_reader_intan__rhd

EXAMPLE_RHD = Path(ndrpath()) / "example_data" / "example.rhd"


@pytest.fixture
def multi(tmp_path: Path) -> dict[str, Path]:
    """A directory holding a two-file set, a lone timestamped file, and a
    file with no timestamp at all."""
    if not EXAMPLE_RHD.exists():  # pragma: no cover - example data is shipped
        pytest.skip(f"example data not found at {EXAMPLE_RHD}")
    plain = tmp_path / "plain.rhd"
    first = tmp_path / "recording_240101_120000.rhd"
    second = tmp_path / "recording_240101_120100.rhd"
    solo_dir = tmp_path / "solo"
    solo_dir.mkdir()
    solo = solo_dir / "recording_240101_120000.rhd"
    for target in (plain, first, second, solo):
        shutil.copy(EXAMPLE_RHD, target)
    return {"plain": plain, "first": first, "second": second, "solo": solo}


class TestDetectFileMode:
    """Mirrors TestIntanRhd.testDetectFileMode."""

    def test_a_file_without_a_timestamp_is_single(self, multi):
        assert detectRHD2000FileMode(multi["plain"]) == "singleFile"

    def test_a_lone_timestamped_file_is_single(self, multi):
        """One file is not a set. This is the boundary the whole feature
        turns on: getting it wrong makes every ordinary recording look
        like a fragment of something larger."""
        assert detectRHD2000FileMode(multi["solo"]) == "singleFile"

    def test_two_siblings_are_multi_from_either_member(self, multi):
        assert detectRHD2000FileMode(multi["first"]) == "multiFile"
        assert detectRHD2000FileMode(multi["second"]) == "multiFile"


class TestGetFileList:
    """Mirrors the file-list half of TestIntanRhd.testMultiFileMode."""

    def test_multifile_lists_the_set_chronologically(self, multi):
        files = getRHD2000FileList(multi["second"], "multiFile")
        assert [Path(f).name for f in files] == [
            "recording_240101_120000.rhd",
            "recording_240101_120100.rhd",
        ]

    def test_singlefile_ignores_siblings(self, multi):
        assert getRHD2000FileList(multi["first"], "singleFile") == [str(multi["first"])]

    def test_an_unknown_mode_is_rejected(self, multi):
        with pytest.raises(ValueError, match="Unknown fileMode"):
            getRHD2000FileList(multi["first"], "bothFiles")


class TestHeaderAggregation:
    """The header describes the whole recording, parsed from the first file."""

    def test_detect_aggregates_a_set(self, multi):
        header = read_Intan_RHD2000_header(multi["first"])
        assert header["fileinfo"]["multifile"]["fileMode"] == "multiFile"
        assert len(header["fileinfo"]["multifile"]["files"]) == 2

    def test_detect_leaves_a_solo_file_alone(self, multi):
        header = read_Intan_RHD2000_header(multi["plain"])
        assert header["fileinfo"]["multifile"]["fileMode"] == "singleFile"
        assert len(header["fileinfo"]["multifile"]["files"]) == 1

    def test_the_headersize_is_shared_across_the_set(self, multi):
        """Acquisition parameters match across a set, so one parse
        characterises the layout -- the assumption the block arithmetic
        rests on."""
        aggregated = read_Intan_RHD2000_header(multi["first"])
        alone = read_Intan_RHD2000_header(multi["plain"])
        assert aggregated["fileinfo"]["headersize"] == alone["fileinfo"]["headersize"]


class TestBlockCounts:
    def test_blocks_are_summed_across_the_set(self, multi):
        single_header = read_Intan_RHD2000_header(multi["plain"])
        _bi, _bpb, _bp, single_blocks, single_file_blocks = Intan_RHD2000_blockinfo(
            multi["plain"], single_header
        )
        multi_header = read_Intan_RHD2000_header(multi["first"])
        _bi2, _bpb2, _bp2, multi_blocks, multi_file_blocks = Intan_RHD2000_blockinfo(
            multi["first"], multi_header
        )
        assert single_file_blocks == [single_blocks]
        assert multi_file_blocks == [single_blocks, single_blocks]
        assert multi_blocks == 2 * single_blocks


class TestReadingAcrossFiles:
    """The point of the feature: a request spanning the set is served."""

    def test_the_whole_recording_is_twice_one_file(self, multi):
        one = read_Intan_RHD2000_datafile(multi["plain"], "", "amp", 1)
        both = read_Intan_RHD2000_datafile(multi["first"], "", "amp", 1)
        assert both.shape[0] == 2 * one.shape[0]
        assert np.array_equal(both[: one.shape[0]], one)
        assert np.array_equal(both[one.shape[0] :], one)

    def test_forcing_singlefile_reads_only_the_named_file(self, multi):
        one = read_Intan_RHD2000_datafile(multi["plain"], "", "amp", 1)
        forced = read_Intan_RHD2000_datafile(multi["first"], "", "amp", 1, fileMode="singleFile")
        assert np.array_equal(forced, one)

    def test_a_window_inside_the_first_file_does_not_reach_the_second(self, multi):
        one = read_Intan_RHD2000_datafile(multi["plain"], "", "amp", 1)
        header = read_Intan_RHD2000_header(multi["plain"])
        rate = header["frequency_parameters"]["amplifier_sample_rate"]
        stop = (one.shape[0] - 1) / rate
        window = read_Intan_RHD2000_datafile(multi["first"], "", "amp", 1, 0.0, stop)
        assert np.array_equal(window, one)

    def test_a_window_spanning_the_boundary_takes_from_both(self, multi):
        """The case that fails if the range arithmetic is off by one."""
        one = read_Intan_RHD2000_datafile(multi["plain"], "", "amp", 1)
        header = read_Intan_RHD2000_header(multi["plain"])
        rate = header["frequency_parameters"]["amplifier_sample_rate"]
        n = one.shape[0]
        start, stop = (n - 5) / rate, (n + 4) / rate
        window = read_Intan_RHD2000_datafile(multi["first"], "", "amp", 1, start, stop)
        assert window.shape[0] == 10
        assert np.array_equal(window[:5], one[n - 5 :])
        assert np.array_equal(window[5:], one[:5])


class TestReaderIntegration:
    """The reader picks a set up without the caller asking for it."""

    def test_several_rhd_files_are_one_epoch(self, multi):
        reader = ndr_reader_intan__rhd()
        filename, _parentdir, _isdir, fileMode = reader.filenamefromepochfiles(
            [str(multi["second"]), str(multi["first"])]
        )
        assert fileMode == "multiFile"
        assert filename == str(multi["first"])

    def test_t0_t1_covers_the_whole_set(self, multi):
        """The span grows by a whole file, which is NOT double the span.

        A recording of N samples spans (N-1)/rate seconds -- the interval
        between first and last sample, not the duration of N samples. So
        two files of N samples span (2N-1)/rate, which exceeds twice
        (N-1)/rate by exactly one sample interval. Asserting a plain
        doubling here fails by 1/rate, and writing the assertion loosely
        enough to absorb that would also absorb a genuine one-sample error
        at the file boundary -- which is precisely the bug worth catching.
        """
        reader = ndr_reader_intan__rhd()
        header = read_Intan_RHD2000_header(multi["plain"])
        rate = header["frequency_parameters"]["amplifier_sample_rate"]

        alone = reader.t0_t1([str(multi["plain"])])[0]
        together = reader.t0_t1([str(multi["first"])])[0]
        one_span = alone[1] - alone[0]
        both_span = together[1] - together[0]

        assert both_span == pytest.approx(2 * one_span + 1 / rate, abs=1e-9)
