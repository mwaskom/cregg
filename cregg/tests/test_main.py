from __future__ import division
import os
import json
from textwrap import dedent
import nose.tools as nt
from .. import main


def test_params():

    try:
        with open("params.py", "w") as fid:
            fid.write(dedent("""
                             foo = dict(bar='hello world',
                                        resp_keys=["a", "b"],
                                        fmri_resp_keys=["1", "2"],
                                        monitor_name="laptop",
                                        fmri_monitor_name="scanner")
                             """))

        p = main.Params("foo")
        assert p.bar == "hello world"

        p.set_by_cmdline(["-s", "cj", "-debug"])
        assert p.subject == "cj"
        assert p.debug
        assert not p.fmri
        assert p.resp_keys == ["a", "b"]
        assert p.monitor_name == "laptop"

        p.to_json("params.json")
        assert os.path.exists("params.json")
        with open("params.json") as fid:
            archive = json.load(fid)
        assert archive["subject"] == "cj"
        assert archive["debug"]

        p.set_by_cmdline(["-fmri"])
        assert p.resp_keys == ["1", "2"]
        assert p.monitor_name == "scanner"

    finally:
        for f in ["params.py", "params.pyc", "params.json"]:
            if os.path.exists(f):
                os.remove(f)


def test_categorical():

    # Very simple tests
    sample = main.categorical([.5, .5])
    assert sample in [0, 1]

    sample = main.categorical([1, 0])
    nt.assert_equal(sample, 0)


def test_subject_state():

    subj = "cj"
    state1 = main.subject_specific_state(subj)
    state2 = main.subject_specific_state(subj)
    assert state1.rand() == state2.rand()
    state3 = main.subject_specific_state("danny")
    assert state1.rand() != state3.rand()


def test_archive_old_version():

    orig = "test_file.txt"
    new = "test_file_1.txt"
    text = "my secret service codename is 'flamingo'"
    try:
        with open(orig, "w") as fid:
            fid.write(text)
        main.archive_old_version(orig)

        assert os.path.exists(new)
        moved_text = open(new).read().strip()
        assert text == moved_text
    finally:
        for f in [orig, new]:
            if os.path.exists(f):
                os.remove(f)
