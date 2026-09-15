"""Direct CUDA implementation of the gated sparse token core; no torch import."""
from __future__ import annotations

import ctypes as c
from pathlib import Path
from typing import Any

import numpy as np

from .cuda_light import Driver, _bind, _check
from .sequence_core import OFFSETS, SIZE, CpuSequence, H, K, V

_MACROS = {"H": H, "V": V, "K": K, **{name.upper().replace("E_", "E"): offset for name, offset in OFFSETS.items()}}
KERNEL = "\n".join(f"#define {name} {value}" for name, value in _MACROS.items())+r'''
__device__ double clip(double value, double limit) { return fmin(limit, fmax(-limit,value)); }
extern "C" __global__ void forward(double* x, const int* parents, int token) {
  if (blockIdx.x) return;
  __shared__ double old[H];
  __shared__ double normalizer;
  for (int i=threadIdx.x;i<H;i+=blockDim.x) old[i]=x[S+i];
  __syncthreads();
  for (int i=threadIdx.x;i<H;i+=blockDim.x) {
    double recurrent=0;
    for (int k=0;k<K;++k) recurrent+=x[R+i*K+k]*old[parents[i*K+k]];
    double proposal=tanh(x[W+i*V+token]+recurrent+x[BC+i]);
    double gate=1.0/(1.0+exp(-(x[G+i*V+token]+x[A+i]*old[i]+x[BG+i])));
    double dc=(1-gate)*(1-proposal*proposal);
    double dg=(old[i]-proposal)*gate*(1-gate);
    double j=gate+dg*x[A+i];
    for (int k=0;k<V;++k) {
      x[EW+i*V+k]=clip(j*x[EW+i*V+k]+(k==token ? dc : 0),4);
      x[EG+i*V+k]=clip(j*x[EG+i*V+k]+(k==token ? dg : 0),4);
    }
    for (int k=0;k<K;++k) x[ER+i*K+k]=clip(j*x[ER+i*K+k]+dc*old[parents[i*K+k]],4);
    x[EA+i]=clip(j*x[EA+i]+dg*old[i],4);
    x[EBC+i]=clip(j*x[EBC+i]+dc,4);
    x[EBG+i]=clip(j*x[EBG+i]+dg,4);
    x[S+i]=gate*old[i]+(1-gate)*proposal;
  }
  __syncthreads();
  for (int k=threadIdx.x;k<V;k+=blockDim.x) {
    double score=0; for (int i=0;i<H;++i) score+=x[O+k*H+i]*x[S+i];
    x[P+k]=score+x[BO+k];
  }
  __syncthreads();
  if (threadIdx.x==0) {
    double maximum=-1e300, sum=0;
    for (int k=0;k<V;++k) maximum=fmax(maximum,x[P+k]);
    for (int k=0;k<V;++k) { x[P+k]=exp(x[P+k]-maximum); sum+=x[P+k]; }
    normalizer=sum;
  }
  __syncthreads();
  for (int k=threadIdx.x;k<V;k+=blockDim.x) x[P+k]/=normalizer;
}
extern "C" __global__ void head(double* x, int target, int core_learning) {
  if (blockIdx.x) return;
  __shared__ double error[V];
  __shared__ double norm;
  for (int k=threadIdx.x;k<V;k+=blockDim.x) error[k]=(k==target ? 1. : 0.)-x[P+k];
  if (threadIdx.x==0) {
    norm=0;
    for (int i=0;i<H;++i) norm+=x[S+i]*x[S+i];
    norm=fmax(1.,norm);
  }
  __syncthreads();
  if (core_learning) {
    for (int i=threadIdx.x;i<H;i+=blockDim.x) {
      double signal=0; for (int k=0;k<V;++k) signal+=x[B+i*V+k]*error[k];
      signal=clip(signal,1);
      for (int k=0;k<V;++k) {
        x[W+i*V+k]=clip(x[W+i*V+k]+clip(.001*signal*x[EW+i*V+k],.02),3);
        x[G+i*V+k]=clip(x[G+i*V+k]+clip(.001*signal*x[EG+i*V+k],.02),3);
      }
      for (int k=0;k<K;++k) x[R+i*K+k]=clip(x[R+i*K+k]+clip(.001*signal*x[ER+i*K+k],.02),3);
      x[A+i]=clip(x[A+i]+clip(.001*signal*x[EA+i],.02),3);
      x[BC+i]=clip(x[BC+i]+clip(.001*signal*x[EBC+i],.02),3);
      x[BG+i]=clip(x[BG+i]+clip(.001*signal*x[EBG+i],.02),3);
    }
  }
  for (int k=threadIdx.x;k<V;k+=blockDim.x) {
    for (int i=0;i<H;++i) x[O+k*H+i]=clip(x[O+k*H+i]+clip(.05*error[k]*x[S+i]/norm,.02),3);
    x[BO+k]=clip(x[BO+k]+clip(.05*error[k],.02),3);
  }
}
'''


class SequenceDriver(Driver):
    def _compile(self, compiler: Path) -> Any:
        # Same installed compiler path and cleanup contract, different source.
        self.builtins_library = c.WinDLL(str((compiler.parent / "nvrtc-builtins64_129.dll").resolve()))
        rtc = c.WinDLL(str(compiler.resolve()))
        pointer = c.c_void_p
        _bind(rtc, "nvrtcCreateProgram", [c.POINTER(pointer), c.c_char_p, c.c_char_p, c.c_int, pointer, pointer])
        _bind(rtc, "nvrtcCompileProgram", [pointer, c.c_int, c.POINTER(c.c_char_p)])
        _bind(rtc, "nvrtcDestroyProgram", [c.POINTER(pointer)])
        for suffix in ("PTX", "ProgramLog"):
            _bind(rtc, f"nvrtcGet{suffix}Size", [pointer, c.POINTER(c.c_size_t)])
            _bind(rtc, f"nvrtcGet{suffix}", [pointer, pointer])
        program = pointer()
        _check(rtc.nvrtcCreateProgram(c.byref(program), KERNEL.encode(), b"sequence.cu", 0, None, None), "create sequence program")
        try:
            options = (c.c_char_p*3)(f"--gpu-architecture={self.architecture}".encode(), b"--fmad=false", b"--std=c++11")
            status = rtc.nvrtcCompileProgram(program, 3, options)
            length = c.c_size_t()
            _check(rtc.nvrtcGetProgramLogSize(program, c.byref(length)), "compile log size")
            log = c.create_string_buffer(length.value)
            _check(rtc.nvrtcGetProgramLog(program, log), "compile log")
            if status:
                raise RuntimeError(f"sequence compile status {status}: {log.value.decode()}")
            _check(rtc.nvrtcGetPTXSize(program, c.byref(length)), "PTX size")
            ptx = c.create_string_buffer(length.value)
            _check(rtc.nvrtcGetPTX(program, ptx), "get PTX")
            return ptx
        finally:
            _check(rtc.nvrtcDestroyProgram(c.byref(program)), "destroy sequence program")


class GpuSequence(CpuSequence):
    def __init__(self, driver: SequenceDriver, seed: int = 15000, *, core_learning: bool = True) -> None:
        if not isinstance(driver, SequenceDriver) or not driver.context:
            raise TypeError("an open SequenceDriver is required")
        super().__init__(seed, core_learning=core_learning)
        self.driver = driver
        self.address = self.parent_address = None
        self._install(self._buffer, self._parents)
        self._buffer = self._parents = None

    def _open(self) -> None:
        super()._open()
        if not self.driver.context:
            raise RuntimeError("sequence CUDA context is closed")

    def _export(self) -> tuple[np.ndarray, np.ndarray]:
        self._open()
        return (self.driver.read(self.address, np.empty(SIZE, dtype=np.float64)),
                self.driver.read(self.parent_address, np.empty((H, K), dtype=np.int32)))

    def _install(self, buffer: np.ndarray, parents: np.ndarray) -> None:
        # Stage both allocations before replacing a live checkpoint/reset state.
        address = self.driver.allocate(np.ascontiguousarray(buffer))
        try:
            parent_address = self.driver.allocate(np.ascontiguousarray(parents))
        except BaseException:
            self.driver.free(address)
            raise
        for previous in (self.address, self.parent_address):
            if previous is not None:
                self.driver.free(previous)
        self.address, self.parent_address = address, parent_address

    def _probabilities(self) -> np.ndarray:
        self._open()
        result = np.empty(V, dtype=np.float64)
        if self.driver.allocations.get(self.address.value) != SIZE*8:
            raise RuntimeError("sequence buffer was released")
        pointer = c.c_uint64(self.address.value+OFFSETS["p"]*8)
        _check(self.driver.driver.cuMemcpyDtoH_v2(result.ctypes.data, pointer, result.nbytes), "copy byte probabilities")
        return result

    def _forward(self, token: int) -> None:
        self.driver.launch("forward", [self.address, self.parent_address, c.c_int(token)], batch=64)

    def _update(self, target: int) -> None:
        self.driver.launch("head", [self.address, c.c_int(target), c.c_int(self.core_learning)], batch=64)

    def close(self) -> None:
        if not self.closed:
            for address in (self.address, self.parent_address):
                if address is not None:
                    self.driver.free(address)
        self.closed = True
