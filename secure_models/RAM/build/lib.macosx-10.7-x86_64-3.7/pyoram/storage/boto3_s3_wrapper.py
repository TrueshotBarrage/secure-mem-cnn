__all__ = ("Boto3S3Wrapper",
           "MockBoto3S3Wrapper")
import os
import shutil

import pyoram

import tqdm
try:
    import boto3
    import botocore
    boto3_available = True
except:                                                # pragma: no cover
    boto3_available = False                            # pragma: no cover

import six
from six.moves import xrange, map

class Boto3S3Wrapper(object):
    """
    A wrapper class for the boto3 S3 service.
    """

    def __init__(self,
                 bucket_name,
                 aws_access_key_id=None,
                 aws_secret_access_key=None,
                 region_name=None):
        if not boto3_available:
            raise ImportError(                         # pragma: no cover
                "boto3 module is required to "         # pragma: no cover
                "use BlockStorageS3 device")           # pragma: no cover

        self._s3 = boto3.session.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name).resource('s3')
        self._bucket = self._s3.Bucket(bucket_name)

    def exists(self, key):
        try:
            self._bucket.Object(key).load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                pass
            else:
                raise e
        else:
            return True
        # It's not a file. Check if it's a "directory".
        for obj in self._bucket.objects.filter(
                Prefix=key+"/"):
            return True
        return False

    def download(self, key):
        try:
            return self._s3.meta.client.get_object(
                Bucket=self._bucket.name,
                Key=key)['Body'].read()
        except botocore.exceptions.ClientError:
            raise IOError("Can not download key: %s"
                          % (key))

    def upload(self, key_block):
        key, block = key_block
        self._bucket.put_object(Key=key, Body=block)

    # Chunk a streamed iterator of which we do not know
    # the size
    def _chunks(self, objs, n=100):
        assert 1 <= n <= 1000 # required by boto3
        objs = iter(objs)
        try:
            while (1):
                chunk = []
                while len(chunk) < n:
                    chunk.append({'Key': six.next(objs).key})
                yield {'Objects': chunk}
        except StopIteration:
            pass
        if len(chunk):
            yield {'Objects': chunk}

    def _del(self, chunk):
        self._bucket.delete_objects(Delete=chunk)
        return len(chunk['Objects'])

    def clear(self, key, threadpool=None):
        objs = self._bucket.objects.filter(Prefix=key+"/")
        if threadpool is not None:
            deliter = threadpool.imap(self._del, self._chunks(objs))
        else:
            deliter = map(self._del, self._chunks(objs))
        with tqdm.tqdm(total=None, #len(objs),
                       desc="Clearing S3 Blocks",
                       unit=" objects",
                       disable=not pyoram.config.SHOW_PROGRESS_BAR) as progress_bar:
            progress_bar.update(n=0)
            for chunksize in deliter:
                progress_bar.update(n=chunksize)

class MockBoto3S3Wrapper(object):
    """
    A mock class for Boto3S3Wrapper that uses the local filesystem and
    treats the bucket name as a directory.

    This class is mainly used for testing, but could potentially be
    used to setup storage locally that is then uploaded to S3 through
    the AWS web portal.
    """

    def __init__(self,
                 bucket_name,
                 aws_access_key_id=None,
                 aws_secret_access_key=None,
                 region_name=None):

        self._bucket_name = os.path.abspath(
            os.path.normpath(bucket_name))

    # called within upload to create directory
    # heirarchy on the fly
    def _makedirs_if_needed(self, key):
        if not os.path.exists(
                os.path.dirname(
                    os.path.join(self._bucket_name, key))):
            os.makedirs(
                os.path.dirname(
                    os.path.join(self._bucket_name, key)))
        assert not os.path.isdir(
            os.path.join(self._bucket_name, key))

    def exists(self, key):
        return os.path.exists(
            os.path.join(self._bucket_name, key))

    def download(self, key):
        with open(os.path.join(self._bucket_name, key), 'rb') as f:
            return f.read()

    def upload(self, key_block):
        key, block = key_block
        self._makedirs_if_needed(key)
        with open(os.path.join(self._bucket_name, key), 'wb') as f:
            f.write(block)

    def clear(self, key, threadpool=None):
        if os.path.exists(
                os.path.join(self._bucket_name, key)):
            if os.path.isdir(
                    os.path.join(self._bucket_name, key)):
                shutil.rmtree(
                    os.path.join(self._bucket_name, key),
                    ignore_errors=True)
            else:
                os.remove(
                    os.path.join(self._bucket_name, key))
