from __future__ import division
import nose.tools as nt
import cregg 


def test_categorical():

    # Very simple tests
    sample = tools.categorical([.5, .5])
    assert sample in [0, 1]

    sample = tools.categorical([1, 0])
    nt.assert_equal(sample, 0)
