"""Small CUDA Driver/NVRTC adapter: no torch import, no host offload or fallback.

Development backend only. Device memory and contexts are explicitly released;
the unchanged NumPy event simulator is the numerical reference.
"""
from __future__ import annotations

import ctypes as c
import math
import os
import sys
from pathlib import Path
from typing import Any

from .gpu_lwoh import Topology, validate_batch

KERNEL = r'''
extern "C" __global__ void forward(
    const int* sources, const double* weights, const int* selected,
    const int* cues, const int* noise, int width, int batch, int delay,
    double threshold, double epsilon, int blank, int trace,
    double* result, double* frames) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  double activation[68] = {0}, last[68] = {0}, previous[69] = {0};
  int present[69] = {0};
  double memory[8] = {0}, latest[8] = {0};
  int written[8] = {0}, mt[8] = {0}, lw[8] = {0}, lt[8] = {0};
  double events=0, touches=0, evaluations=0, calls=0, tail=0, expired=0;
  for (int tick=0; tick<delay+4; ++tick) {
    double updated[65]; int evaluated[65], emitted[65];
    for (int j=0; j<65; ++j) {
      double total=0, correction=0; int any=0;
      for (int k=0; k<width; ++k) {
        int src=sources[j*width+k];
        if (present[src]) {
          ++touches; any=1;
          double x=weights[j*width+k]*previous[src];
          double next=total+x;
          correction += fabs(total)>=fabs(x) ? (total-next)+x : (x-next)+total;
          total=next;
        }
      }
      evaluated[j]=any || (tick==delay+3 && j==64);
      updated[j]=tanh(total+correction);
      emitted[j]=evaluated[j] && (fabs(updated[j]-last[j+3])>=threshold
                                  || (tick==delay+3 && j==64));
      if (evaluated[j]) { activation[j+3]=updated[j]; ++evaluations; }
      if (emitted[j]) { last[j+3]=updated[j]; ++events; }
      if (blank && tick>=193 && tick<=256 && j<64) tail+=emitted[j];
    }
    for (int k=0; k<8; ++k) {
      int j=selected[k];
      if (evaluated[j]) {
        ++calls;
        if (!written[k]) { memory[k]=activation[j+3]; mt[k]=tick; written[k]=1; }
        latest[k]=activation[j+3]; lt[k]=tick; lw[k]=1;
      }
      if (blank && tick==256 && written[k] && tick-mt[k]<=32)
        expired=fmax(expired, fabs(memory[k]));
    }
    for (int j=0; j<69; ++j) { previous[j]=0; present[j]=0; }
    double external[3] = {0,0,0};
    if (tick==0) external[0]=cues[b];
    if (!blank && tick>=1 && tick<=delay) external[1]=noise[b*delay+tick-1];
    if (tick==delay+1) external[2]=1;
    for (int j=0; j<3; ++j) {
      previous[j]=external[j]; present[j]=fabs(external[j])>=epsilon;
      if (present[j]) { activation[j]=external[j]; last[j]=external[j]; ++events; }
    }
    for (int j=0; j<65; ++j) { previous[j+3]=updated[j]; present[j+3]=emitted[j]; }
    if (trace) {
      int offset=(b*(delay+4)+tick)*133;
      for (int j=0; j<65; ++j) frames[offset+j]=activation[j+3];
      for (int j=0; j<68; ++j) frames[offset+65+j]=present[j];
    }
  }
  int offset=b*24; double writes=0;
  for (int k=0; k<8; ++k) {
    result[offset+k]=(written[k] && delay+3-mt[k]<=32) ? memory[k] : 0;
    result[offset+8+k]=(lw[k] && delay+3-lt[k]<=32) ? latest[k] : 0;
    writes+=written[k];
  }
  result[offset+16]=activation[67]; result[offset+17]=events;
  result[offset+18]=touches; result[offset+19]=evaluations;
  result[offset+20]=calls; result[offset+21]=writes;
  result[offset+22]=tail; result[offset+23]=expired;
}
extern "C" __global__ void head(double* theta, const double* feature,
                                 int update, int target, double* result) {
  double sum=0, norm=0;
  for (int k=0; k<8; ++k) { sum+=theta[k]*feature[k]; norm+=feature[k]*feature[k]; }
  double q=tanh(sum); result[0]=q; result[1]=q>=0 ? 1 : -1;
  for (int k=0; k<8; ++k) {
    double delta=update ? fmin(.05, fmax(-.05, .5*(target-q)*(1-q*q)*feature[k]/fmax(1.,norm))) : 0;
    if (update) theta[k]=fmin(3., fmax(-3., theta[k]+delta));
    result[k+2]=delta; result[k+10]=theta[k];
  }
}
'''


def _bind(library: Any, name: str, args: list[Any]) -> Any:
    function = getattr(library, name)
    function.argtypes, function.restype = args, c.c_int
    return function


def _check(status: int, label: str) -> None:
    if status:
        raise RuntimeError(f"{label} returned CUDA/NVRTC status {status}")


class Driver:
    """One private context, bounded allocations; used only on its owning thread."""

    def __init__(self, compiler: Path) -> None:
        if os.name != "nt":
            raise RuntimeError("this adapter requires the Windows CUDA driver")
        if not compiler.is_file() or compiler.suffix != ".dll":
            raise ValueError("provide the installed NVRTC DLL path")
        self.context, self.module = c.c_void_p(), c.c_void_p()
        self.allocations: dict[int, int] = {}
        self.peak_allocated = 0
        self.directory = os.add_dll_directory(str(compiler.parent.resolve()))
        self.driver = d = c.WinDLL("nvcuda.dll")
        pointer, integer, size, device_pointer = c.c_void_p, c.c_int, c.c_size_t, c.c_uint64
        signatures = {
            "cuInit": [c.c_uint], "cuDeviceGet": [c.POINTER(integer), integer],
            "cuDeviceComputeCapability": [c.POINTER(integer), c.POINTER(integer), integer],
            "cuCtxCreate_v2": [c.POINTER(pointer), c.c_uint, integer],
            "cuCtxDestroy_v2": [pointer], "cuCtxSynchronize": [],
            "cuMemGetInfo_v2": [c.POINTER(size), c.POINTER(size)],
            "cuMemAlloc_v2": [c.POINTER(device_pointer), size], "cuMemFree_v2": [device_pointer],
            "cuMemcpyHtoD_v2": [device_pointer, pointer, size],
            "cuMemcpyDtoH_v2": [pointer, device_pointer, size],
            "cuModuleLoadData": [c.POINTER(pointer), pointer], "cuModuleUnload": [pointer],
            "cuModuleGetFunction": [c.POINTER(pointer), pointer, c.c_char_p],
            "cuLaunchKernel": [pointer] + [c.c_uint]*7 + [pointer, c.POINTER(pointer), c.POINTER(pointer)],
        }
        for name, args in signatures.items():
            _bind(d, name, args)
        try:
            _check(d.cuInit(0), "cuInit")
            device, major, minor = integer(), integer(), integer()
            _check(d.cuDeviceGet(c.byref(device), 0), "cuDeviceGet")
            _check(d.cuDeviceComputeCapability(c.byref(major), c.byref(minor), device), "device capability")
            self.architecture = f"compute_{major.value}{minor.value}"
            _check(d.cuCtxCreate_v2(c.byref(self.context), 0, device), "cuCtxCreate")
            free, total = size(), size()
            _check(d.cuMemGetInfo_v2(c.byref(free), c.byref(total)), "cuMemGetInfo")
            self.device_free_before, self.device_total = free.value, total.value
            if free.value < 256 * 1024**2:
                raise MemoryError("less than 256 MiB device headroom")
            ptx = self._compile(compiler)
            _check(d.cuModuleLoadData(c.byref(self.module), c.cast(ptx, pointer)), "cuModuleLoadData")
            self.ptx_sha256 = __import__("hashlib").sha256(ptx.raw).hexdigest()
            self.functions = {}
            for name in ("forward", "head"):
                function = pointer()
                _check(d.cuModuleGetFunction(c.byref(function), self.module, name.encode()), "cuModuleGetFunction")
                self.functions[name] = function
        except BaseException:
            self.close()
            raise

    def _compile(self, compiler: Path) -> Any:
        # NVRTC's internal LoadLibrary does not honor Python's DLL search flags.
        # Preload its installed companion by absolute path, never alter global PATH.
        builtins = compiler.parent / "nvrtc-builtins64_129.dll"
        self.builtins_library = c.WinDLL(str(builtins.resolve()))
        rtc = c.WinDLL(str(compiler.resolve()))
        p = c.c_void_p
        _bind(rtc, "nvrtcCreateProgram", [c.POINTER(p), c.c_char_p, c.c_char_p, c.c_int, p, p])
        _bind(rtc, "nvrtcCompileProgram", [p, c.c_int, c.POINTER(c.c_char_p)])
        _bind(rtc, "nvrtcDestroyProgram", [c.POINTER(p)])
        for suffix in ("PTX", "ProgramLog"):
            _bind(rtc, f"nvrtcGet{suffix}Size", [p, c.POINTER(c.c_size_t)])
            _bind(rtc, f"nvrtcGet{suffix}", [p, p])
        program = p()
        _check(rtc.nvrtcCreateProgram(c.byref(program), KERNEL.encode(), b"lwoh.cu", 0, None, None), "nvrtcCreateProgram")
        try:
            options = (c.c_char_p * 3)(f"--gpu-architecture={self.architecture}".encode(), b"--fmad=false", b"--std=c++11")
            status = rtc.nvrtcCompileProgram(program, 3, options)
            size = c.c_size_t()
            _check(rtc.nvrtcGetProgramLogSize(program, c.byref(size)), "NVRTC log size")
            log = c.create_string_buffer(size.value)
            _check(rtc.nvrtcGetProgramLog(program, log), "NVRTC log")
            if status:
                raise RuntimeError(f"NVRTC compile failed ({status}): {log.value.decode()}")
            _check(rtc.nvrtcGetPTXSize(program, c.byref(size)), "NVRTC PTX size")
            ptx = c.create_string_buffer(size.value)
            _check(rtc.nvrtcGetPTX(program, ptx), "NVRTC PTX")
            return ptx
        finally:
            _check(rtc.nvrtcDestroyProgram(c.byref(program)), "nvrtcDestroyProgram")

    def allocate(self, array: Any) -> c.c_uint64:
        if not self.context:
            raise RuntimeError("CUDA context is closed")
        if sum(self.allocations.values()) + array.nbytes > 16 * 1024**2:
            raise MemoryError("16 MiB explicit CUDA allocation cap exceeded")
        address = c.c_uint64()
        _check(self.driver.cuMemAlloc_v2(c.byref(address), array.nbytes), "cuMemAlloc")
        self.allocations[address.value] = array.nbytes
        self.peak_allocated = max(self.peak_allocated, sum(self.allocations.values()))
        try:
            _check(self.driver.cuMemcpyHtoD_v2(address, array.ctypes.data, array.nbytes), "cuMemcpyHtoD")
        except BaseException:
            self.free(address)
            raise
        return address

    def read(self, address: c.c_uint64, array: Any) -> Any:
        if self.allocations.get(address.value) != array.nbytes:
            raise ValueError("device/host allocation size mismatch")
        _check(self.driver.cuMemcpyDtoH_v2(array.ctypes.data, address, array.nbytes), "cuMemcpyDtoH")
        return array

    def launch(self, name: str, args: list[Any], batch: int = 1) -> None:
        parameters = (c.c_void_p * len(args))(*(c.cast(c.byref(arg), c.c_void_p) for arg in args))
        _check(self.driver.cuLaunchKernel(self.functions[name], 1, 1, 1, batch, 1, 1, 0, None, parameters, None), name)
        _check(self.driver.cuCtxSynchronize(), "cuCtxSynchronize")

    def free(self, address: c.c_uint64) -> None:
        if address.value in self.allocations:
            _check(self.driver.cuMemFree_v2(address), "cuMemFree")
            del self.allocations[address.value]

    def close(self) -> None:
        for address in tuple(self.allocations):
            self.free(c.c_uint64(address))
        if self.module:
            _check(self.driver.cuModuleUnload(self.module), "cuModuleUnload")
            self.module = c.c_void_p()
        if self.context:
            _check(self.driver.cuCtxDestroy_v2(self.context), "cuCtxDestroy")
            self.context = c.c_void_p()
        self.directory.close()


def default_compiler() -> Path:
    return Path(sys.prefix) / "Lib/site-packages/torch/lib/nvrtc64_120_0.dll"


class LightForward:
    def __init__(self, driver: Driver, topology: Topology) -> None:
        self.driver, self.spec = driver, topology

    def run(self, cues: Any, noise: Any, *, trace: bool = False, tail: bool = False) -> dict[str, Any]:
        import numpy as np
        delay = validate_batch(cues, noise)
        if tail and (delay != 256 or any(cue is None for cue in cues)):
            raise ValueError("tail requires visible bipolar cues and exactly 256 blank ticks")
        if type(trace) is not bool or type(tail) is not bool:
            raise TypeError("trace and tail must be booleans")
        arrays = [np.asarray(self.spec.sources, dtype=np.int32), np.asarray(self.spec.weights, dtype=np.float64),
                  np.asarray(self.spec.selected, dtype=np.int32), np.asarray([0 if x is None else x for x in cues], dtype=np.int32),
                  np.asarray(noise, dtype=np.int32) if delay else np.zeros(1, dtype=np.int32),
                  np.zeros((len(cues), 24), dtype=np.float64),
                  np.zeros((len(cues), delay+4, 133) if trace else (1,), dtype=np.float64)]
        addresses = []
        try:
            for array in arrays:
                addresses.append(self.driver.allocate(array))
            self.driver.launch("forward", addresses[:5] + [c.c_int(len(self.spec.sources[0])), c.c_int(len(cues)),
                               c.c_int(delay), c.c_double(self.spec.threshold), c.c_double(self.spec.epsilon),
                               c.c_int(tail), c.c_int(trace)] + addresses[5:], len(cues))
            result = self.driver.read(addresses[5], arrays[5])
            if not np.isfinite(result).all():
                raise FloatingPointError("non-finite CUDA output")
            frames = self.driver.read(addresses[6], arrays[6]) if trace else None
            return {"tick": delay+3, "lwoh": result[:, :8], "latest": result[:, 8:16], "output": result[:, 16],
                    **{name: result[:, index].astype(np.int64) for index, name in enumerate(
                        ("events", "touches", "evaluations", "observer_calls", "writes", "tail_emissions"), 17)},
                    "expired_abs_max": result[:, 23], "frames": frames}
        finally:
            for address in reversed(addresses):
                self.driver.free(address)


class LightHead:
    def __init__(self, driver: Driver) -> None:
        import numpy as np
        self.driver = driver
        self.address = driver.allocate(np.zeros(8, dtype=np.float64))
        self.predictions = self.updates = 0
        self.pending: tuple[int, Any] | None = None

    def _call(self, feature: Any, update: bool, target: int) -> Any:
        import numpy as np
        feature = np.asarray(feature, dtype=np.float64)
        if feature.shape != (8,) or not np.isfinite(feature).all():
            raise ValueError("head needs eight finite feature values")
        result = np.zeros(18, dtype=np.float64)
        addresses = []
        try:
            for array in (feature.copy(), result):
                addresses.append(self.driver.allocate(array))
            self.driver.launch("head", [self.address, addresses[0], c.c_int(update), c.c_int(target), addresses[1]])
            self.driver.read(addresses[1], result)
            if not all(math.isfinite(x) for x in result):
                raise FloatingPointError("non-finite head result")
            return result
        finally:
            for address in reversed(addresses):
                self.driver.free(address)

    @property
    def theta(self) -> Any:
        import numpy as np
        return self.driver.read(self.address, np.zeros(8, dtype=np.float64))

    def predict(self, feature: Any) -> tuple[float, int]:
        row = self._call(feature, False, 0)
        return float(row[0]), int(row[1])

    def predict_for_update(self, feature: Any) -> tuple[int, float, int]:
        import numpy as np
        if self.pending is not None:
            raise RuntimeError("consume pending target before next prediction")
        score, prediction = self.predict(feature)
        token = self.predictions
        self.pending = token, np.asarray(feature, dtype=np.float64).copy()
        self.predictions += 1
        return token, score, prediction

    def reveal_target(self, token: int, target: int) -> Any:
        if type(target) is not int or target not in (-1, 1):
            raise ValueError("target must be a built-in bipolar integer")
        if type(token) is not int or self.pending is None or self.pending[0] != token:
            raise RuntimeError("target does not match pending prediction")
        row = self._call(self.pending[1], True, target)
        self.pending = None
        self.updates += 1
        return row[2:10]

    def close(self) -> None:
        self.driver.free(self.address)
