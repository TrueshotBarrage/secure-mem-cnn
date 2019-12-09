PyORAM
======

.. image:: https://travis-ci.org/ghackebeil/PyORAM.svg?branch=master
    :target: https://travis-ci.org/ghackebeil/PyORAM

.. image:: https://ci.appveyor.com/api/projects/status/1tpnf7fr0qthrwxx/branch/master?svg=true
    :target: https://ci.appveyor.com/project/ghackebeil/PyORAM?branch=master

.. image:: https://codecov.io/github/ghackebeil/PyORAM/coverage.svg?branch=master
    :target: https://codecov.io/github/ghackebeil/PyORAM?branch=master

.. image:: https://img.shields.io/pypi/v/PyORAM.svg
    :target: https://pypi.python.org/pypi/PyORAM/

Python-based Oblivious RAM (PyORAM) is a collection of
Oblivious RAM algorithms implemented in Python. This package
serves to enable rapid prototyping and testing of new ORAM
algorithms and ORAM-based applications tailored for the
cloud-storage setting. PyORAM is written to support as many
Python versions as possible, including Python 2.7+, Python
3.4+, and PyPy 2.6+.

This software is copyright (c) by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).

This software is released under the MIT software license.
This license, including disclaimer, is available in the 'LICENSE' file.

This work was funded by the Privacy Enhancing Technologies
project under the guidance of Professor `Attila Yavuz
<https://web.engr.oregonstate.edu/~yavuza>`_ at Oregon State
University.

Why Python?
-----------

This project is meant for research. It is provided mainly as
a tool for other researchers studying the applicability of
ORAM to the cloud-storage setting. In such a setting, we
observe that network latency far outweighs any overhead
introduced from switching to an interpreted language such as
Python (as opposed to C++ or Java). Thus, our hope is that
by providing a Python-based library of ORAM tools, we will
enable researchers to spend more time prototyping new and
interesting ORAM applications and less time fighting with a
compiler or chasing down segmentation faults.

Installation
------------

To install the latest release of PyORAM, simply execute::

  $ pip install PyORAM

To install the trunk version of PyORAM, first clone the repository::

  $ git clone https://github.com/ghackebeil/PyORAM.git

Next, enter the directory where PyORAM has been cloned and run setup::

  $ python setup.py install

If you are a developer, you should instead install using::

  $ pip install -e .
  $ pip install nose2 unittest

Installation Tips
-----------------

* OS X users are recommended to work with the `homebrew
  <http://brew.sh/>`_ version of Python2 or Python3. If you
  must use the default system Python, then the best thing to
  do is create a virtual environment and install PyORAM into
  that. The process of creating a virtual environment that is
  stored in the PyORAM directory would look something like::

    $ sudo pip install virtualenv
    $ cd <PyORAM-directory>
    $ virtualenv local_python2.7

  If you had already attempted to install PyORAM into the
  system Python and encountered errors, it may be necessary
  to delete the directories :code:`build` and :code:`dist`
  from the current directory using the command::

    $ sudo rm -rf build dist

  Once this virtual environment has been successfully
  created, you can *activate* it using the command::

    $ . local_python2.7/bin/activate

  Then, proceed with the normal installation steps to
  install PyORAM into this environment. Note that you must
  *activate* this environment each time you open a new
  terminal if PyORAM is installed in this way. Also, note
  that use of the :code:`sudo` command is no longer
  necessary (and should be avoided) once a virtual
  environment is activated in the current shell.

* If you have trouble installing the cryptography package
  on OS X with PyPy: `stackoverflow <https://stackoverflow.com/questions/36662704/fatal-error-openssl-e-os2-h-file-not-found-in-pypy/36706513#36706513>`_.

* If you encounter the dreaded "unable to find
  vcvarsall.bat" error when installing packages with C
  extensions through pip on Windows: `blog post <https://blogs.msdn.microsoft.com/pythonengineering/2016/04/11/unable-to-find-vcvarsall-bat>`_.

Tools Available (So Far)
------------------------

Encrypted block storage
~~~~~~~~~~~~~~~~~~~~~~~

* The basic building block for any ORAM implementation.

* Available storage interfaces include:

  - local storage using a file, a memory-mapped file, or RAM

    + Dropbox

  - cloud storage using SFTP (requires SSH access to a server)

    + Amazon EC2

    + Microsoft Azure

    + Google Cloud Platform

  - cloud storage using Amazon Simple Storage Service (S3)

* See Examples:

  - examples/encrypted_storage_ram.py

  - examples/encrypted_storage_mmap.py

  - examples/encrypted_storage_file.py

  - examples/encrypted_storage_sftp.py

  - examples/encrypted_storage_s3.py

Path ORAM
~~~~~~~~~

* Reference: `Stefanov et al. <http://arxiv.org/abs/1202.5150v3>`_

* Generalized to work over k-kary storage heaps. Default
  settings use a binary storage heap and bucket size
  parameter set to 4. Using a k-ary storage heap can reduce
  the access cost; however, stash size behavior has not been
  formally analyzed in this setting.

* Tree-Top caching can be used to reduce data transmission
  per access as well as reduce access latency by exploiting
  parallelism across independent sub-heaps below the last
  cached heap level.

* See Examples:

  -  examples/path_oram_ram.py

  - examples/path_oram_mmap.py

  - examples/path_oram_file.py

  - examples/path_oram_sftp.py

  - examples/path_oram_s3.py

Performance Tips
----------------

Setup Storage Locally
~~~~~~~~~~~~~~~~~~~~~

Storage schemes such as BlockStorageFile ("file"), BlockStorageMMap
("mmap"), BlockStorageRAM ("ram"), and BlockStorageSFTP ("sftp") all
employ the same underlying storage format. Thus, an oblivious storage
scheme can be initialized locally and then transferred to an external
storage location and accessed via BlockStorageSFTP using SSH login
credentials. See the following pair of files for an example of this:

* examples/path_oram_sftp_setup.py

* examples/path_oram_sftp_test.py

BlockStorageS3 ("s3") employs a different format whereby the
underlying blocks are stored in separate "file" objects.
This design is due to the fact that the Amazon S3 API does
not allow modifications to a specific byte range within a
file, but instead requires that the entire modified file
object be re-uploaded. Thus, any efficient block storage
scheme must use separate "file" objects for each block.

Tree-Top Caching
~~~~~~~~~~~~~~~~

For schemes that employ a storage heap (such as Path ORAM),
tree-top caching provides the ability to parallelize I/O
operations across the independent sub-heaps below the last
cached heap level. The default behavior of this
implementation of Path ORAM, for instance, caches the top
three levels of the storage heap in RAM, which creates eight
independent sub-heaps across which write operations can be
asynchronous.

If the underlying storage is being accessed through SFTP, the
tree-top cached storage heap will attempt to open an
independent SFTP session for each sub-heap using the same
SSH connection. Typically, the maximum number of allowable
sessions associated with a single SSH connection is limited
by the SSH server. For instance, the default maximum number
of sessions allowed by a server using OpenSSH is 10. Thus,
increasing the number of cached levels beyond 3 when using
a binary storage heap will attempt to generate 16 or more SFTP
sessions and result in an error such as::

  paramiko.ssh_exception.ChannelException: (1, 'Administratively prohibited')

There are two options for avoiding this error:

1. If you have administrative privileges on the server, you
   can increase the maximum number of allowed sessions for a
   single SSH connection. For example, to set the maximum
   allowed sessions to 128 on a server using OpenSSH, one
   would set::

     MaxSessions 128

   in :code:`/etc/ssh/sshd_config`, and then run the
   command :code:`sudo service ssh restart`.

2. You can limit the number of concurrent devices that will
   be created by setting the concurrency level to something
   below the last cached level using the
   :code:`concurrency_level` keyword. For example, the
   settings :code:`cached_levels=5` and
   :code:`concurrency_level=0` would cache the top 5 levels
   of the storage heap locally, but all external I/O
   operations would take place through a single storage
   device (e.g., using 1 SFTP session).
