from __future__ import division
import os
import json
import nose.tools as nt
from .. import main 

def test_params():

    try:
        with open("params.py", "w") as fid:
            fid.write("foo = dict(bar='hello world')")

        p = main.Params("foo")
        assert p.bar == "hello world"

        p.set_by_cmdline(["-s", "cj", "-debug"])
        assert p.subject == "cj"
        assert p.debug
        assert not p.fmri

        p.to_json("params.json")
        assert os.path.exists("params.json")
        with open("params.json") as fid:
            archive = json.load(fid)
        assert archive["subject"] == "cj"
        assert archive["debug"]

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
