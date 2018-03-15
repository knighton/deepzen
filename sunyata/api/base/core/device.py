from contextlib import contextmanager


class Device(object):
    def __init__(self, device_id, type, name):
        assert isinstance(device_id, int)
        assert 0 <= device_id
        assert isinstance(type, str)
        assert type
        assert isinstance(name, str)
        assert name
        self.device_id = device_id
        self.type = type
        self.name = name


class CPU(Device):
    def __init__(self, device_id):
        type = 'cpu'
        name = type
        super().__init__(device_id, type, name)


class GPU(Device):
    def __init__(self, device_id, gpu_id):
        assert isinstance(gpu_id, int)
        assert 0 <= gpu_id
        type = 'gpu'
        name = '%s:%d' % (type, gpu_id)
        super().__init__(device_id, type, name)
        self.gpu_id = gpu_id


class BaseDeviceAPI(object):
    def __init__(self, num_gpus, home_device=None):
        self._devices = []

        cpu = CPU(0)
        self._devices.append(cpu)
        self._cpu = cpu

        self._gpus = []
        for gpu_id in range(num_gpus):
            device_id = gpu_id + 1
            gpu = GPU(device_id, gpu_id)
            self._devices.append(gpu)
            self._gpus.append(gpu)

        self._name2device = {}
        for device in self._devices:
            self._name2device[device.name] = device

        if home_device is None:
            device = self._devices[1 if num_gpus else 0]
        elif isinstance(home_device, int):
            device = self._devices[home_device]
        elif isinstance(home_device, str):
            device = self._name2device[home_device]
        else:
            assert False
        self._device_scopes = [device]

    def devices(self):
        return self._devices

    def get_device(self, x=None):
        if x is None:
            return self._device_scopes[-1]
        elif isinstance(x, Device):
            pass
        elif isinstance(x, int):
            x = self._devices[x]
        elif isinstance(x, str):
            x = self._name2device[x]
        else:
            assert False
        return x

    def device(self, x=None):
        raise NotImplementedError

    def cpu(self):
        return self._cpu

    def gpus(self):
        return self._gpus

    def get_gpu(self, x=None):
        if x is None:
            x = self._device_scopes[-1]
            assert isinstance(x, GPU)
        elif isinstance(x, GPU):
            pass
        elif isinstance(x, int):
            x = self._gpus[x]
        elif isinstance(x, str):
            x = self._name2device[x]
            assert isinstance(x, GPU)
        else:
            assert False
        return x

    def gpu(self, x=None):
        raise NotImplementedError

    def set_home_device(self, x):
        self._device_scopes[0] = self.device(x)

    @contextmanager
    def device_scope(self, x):
        device = self.device(x)
        self._device_scopes.append(device)
        yield
        self._device_scopes.pop()
