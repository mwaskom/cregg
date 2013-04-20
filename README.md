Tools for Psychology Experiments
================================

Contains a set of reusable classes and functions that support
the collection of data from experimental subjects.

Some of these tools assume a tightly integrated design ecosystem; others will
be more generally useful. At the moment these are not separated out very well.

Most of this code assumes the use of [PsychoPy](http://www.psychopy.org/)
for controlling the actual stimulus presentation,

The [moss](https://github.com/mwaskom/moss) package also includes some
functions that were written to facilitate experimental design; they
may be merged here eventually.

It is possible that the scope will be expanded in the future to include
utilities inteded to support web-based experiments.

Installation
------------

$ python setup.py install

Testing
-------

$ nosetests

*Note:* The test coverage for these tools is considerably lighter than
would be ideal. Tread carefully.

Development
-----------

https://github.com/mwaskom/cregg
