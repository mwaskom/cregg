from __future__ import division
import nose.tools as nt
from .. import main 


def test_categorical():

    # Very simple tests
    sample = main.categorical([.5, .5])
    assert sample in [0, 1]

    sample = main.categorical([1, 0])
    nt.assert_equal(sample, 0)
