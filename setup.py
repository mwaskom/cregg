#! /usr/bin/env python
#
# Copyright (C) 2012 Michael Waskom <mwaskom@stanford.edu>

descr = """Cregg: assorted utilities for psychology experiments"""

import os


DISTNAME = 'cregg'
DESCRIPTION = descr
MAINTAINER = 'Michael Waskom'
MAINTAINER_EMAIL = 'mwaskom@stanford.edu'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mwaskom/cregg'
VERSION = '0.1.dev'

from numpy.distutils.core import setup


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        download_url=DOWNLOAD_URL,
    )
