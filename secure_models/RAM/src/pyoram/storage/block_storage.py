__all__ = ('BlockStorageTypeFactory',)

import logging

log = logging.getLogger("pyoram")

def BlockStorageTypeFactory(storage_type_name):
    if storage_type_name in BlockStorageTypeFactory._registered_devices:
        return BlockStorageTypeFactory.\
            _registered_devices[storage_type_name]
    else:
        raise ValueError(
            "BlockStorageTypeFactory: Unsupported storage "
            "type: %s" % (storage_type_name))
BlockStorageTypeFactory._registered_devices = {}

def _register_device(name, type_):
    if name in BlockStorageTypeFactory._registered_devices:
        raise ValueError("Can not register block storage device type "
                         "with name '%s'. A device type is already "
                         "registered with that name." % (name))
    if not issubclass(type_, BlockStorageInterface):
        raise TypeError("Can not register block storage device type "
                        "'%s'. The device must be a subclass of "
                        "BlockStorageInterface" % (type_))
    BlockStorageTypeFactory._registered_devices[name] = type_
BlockStorageTypeFactory.register_device = _register_device

class BlockStorageInterface(object):

    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()

    #
    # Abstract Interface
    #

    def clone_device(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover

    @classmethod
    def compute_storage_size(cls, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    @classmethod
    def setup(cls, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover

    @property
    def header_data(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    @property
    def block_count(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    @property
    def block_size(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    @property
    def storage_name(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover

    def update_header_data(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    def close(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    def read_blocks(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    def yield_blocks(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    def read_block(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    def write_blocks(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    def write_block(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover

    @property
    def bytes_sent(self):
        raise NotImplementedError                      # pragma: no cover
    @property
    def bytes_received(self):
        raise NotImplementedError                      # pragma: no cover
