import os
import re
import subprocess
import setuptools
import importlib

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


def get_nvcc_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)
        output = output.decode("utf-8")
        match = re.search(r"release\s+(\d+\.\d+)", output)
        if match:
            return match.group(1)
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


if __name__ == '__main__':
    libraries = []
    use_nixl = False
    disable_nvshmem = False

    nixl_lib_path = os.getenv('NIXL_LIB_PATH', None)
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    nvshmem_host_lib = 'libnvshmem_host.so'

    if nixl_lib_path and os.path.exists(nixl_lib_path):
        use_nixl = True
        if nvshmem_dir:
            print('Warning: Both NIXL_LIB_PATH and NVSHMEM_DIR are set. Preferring NIXL.\n')
        disable_nvshmem = True
    elif nvshmem_dir is None:
        try:
            nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
            nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
            import nvidia.nvshmem as nvshmem  # noqa: F401
        except (ModuleNotFoundError, AttributeError, IndexError):
            print(
                'Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. All internode and low-latency features are disabled\n'
            )
            disable_nvshmem = True

    if not use_nixl and not disable_nvshmem:
        assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'

    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    sources = ['csrc/deep_ep.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu']
    include_dirs = ['csrc/']
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = ['-lcuda']

    # NVCC 12.9 workaround for UCX compilation bug
    nvcc_version = get_nvcc_version()
    if nvcc_version == '12.9':
        cxx_flags.append('-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE')
        nvcc_flags.append('-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE')

    if use_nixl:
        print(f'Building with NIXL backend (NIXL_LIB_PATH={nixl_lib_path})')
        cxx_flags.append('-DUSE_NIXL')
        nvcc_flags.append('-DUSE_NIXL')
        cxx_flags.append('-DDISABLE_NVSHMEM')
        nvcc_flags.append('-DDISABLE_NVSHMEM')

        nixl_include_paths_str = os.getenv('NIXL_INCLUDE_PATHS', '')
        nixl_include_paths = [p for p in nixl_include_paths_str.split(':') if p]

        # Auto-detect DOCA GPU headers (needed by UCX GDAKI -> nixl_device.cuh chain)
        doca_home = os.getenv('DOCA_HOME', '/workspace/doca/build/install')
        doca_search_paths = [
            os.path.join(doca_home, 'include'),
            os.path.join(doca_home, 'lib', 'x86_64-linux-gnu', 'doca', 'include'),
            '/opt/mellanox/doca/include',
            '/opt/mellanox/doca/lib/x86_64-linux-gnu/doca/include',
        ]
        for dp in doca_search_paths:
            if os.path.isfile(os.path.join(dp, 'doca_gpunetio_dev_verbs_qp.cuh')):
                if dp not in nixl_include_paths:
                    nixl_include_paths.append(dp)
                    print(f'Found DOCA GPU headers at: {dp}')
                break
        else:
            # Brute-force search as last resort
            import subprocess
            try:
                result = subprocess.run(['find', doca_home, '-name', 'doca_gpunetio_dev_verbs_qp.cuh', '-type', 'f'],
                                        capture_output=True, text=True, timeout=10)
                if result.stdout.strip():
                    doca_gpu_dir = os.path.dirname(result.stdout.strip().split('\n')[0])
                    if doca_gpu_dir not in nixl_include_paths:
                        nixl_include_paths.append(doca_gpu_dir)
                        print(f'Found DOCA GPU headers at: {doca_gpu_dir}')
                else:
                    print(f'WARNING: doca_gpunetio_dev_verbs_qp.cuh not found under {doca_home}')
                    print(f'  Trying /opt/mellanox/doca ...')
                    result = subprocess.run(['find', '/opt/mellanox/doca', '-name', 'doca_gpunetio_dev_verbs_qp.cuh', '-type', 'f'],
                                            capture_output=True, text=True, timeout=10)
                    if result.stdout.strip():
                        doca_gpu_dir = os.path.dirname(result.stdout.strip().split('\n')[0])
                        if doca_gpu_dir not in nixl_include_paths:
                            nixl_include_paths.append(doca_gpu_dir)
                            print(f'Found DOCA GPU headers at: {doca_gpu_dir}')
                    else:
                        print(f'WARNING: doca_gpunetio_dev_verbs_qp.cuh not found anywhere. CUDA kernels may fail to compile.')
            except Exception as e:
                print(f'WARNING: Could not search for DOCA GPU headers: {e}')

        print(f'NIXL include paths: {nixl_include_paths}')
        include_dirs.extend(nixl_include_paths)

        library_dirs.append(nixl_lib_path)
        library_dirs.append(os.path.join(nixl_lib_path, 'core'))
        library_dirs.append(os.path.join(nixl_lib_path, 'plugins'))
        libraries.append('nixl')

        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu'])

        link_flags = [f'-L{lib_dir}' for lib_dir in library_dirs]
        rpath_flags = [f'-Wl,-rpath,{lib_dir}' for lib_dir in library_dirs]
        nvcc_dlink = ['-dlink'] + link_flags + ['-lnixl']
        extra_link_args.extend(['-lnixl'] + rpath_flags)

        if os.getenv('ENABLE_DEBUG_LOGS', '0') == '1':
            print('Debug logs enabled')
            cxx_flags.append('-DENABLE_DEBUG_LOGS')
            nvcc_flags.append('-DENABLE_DEBUG_LOGS')

    elif disable_nvshmem:
        cxx_flags.append('-DDISABLE_NVSHMEM')
        nvcc_flags.append('-DDISABLE_NVSHMEM')
    else:
        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu'])
        include_dirs.extend([f'{nvshmem_dir}/include'])
        library_dirs.extend([f'{nvshmem_dir}/lib'])
        nvcc_dlink.extend(['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem_device'])
        extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_dir}/lib'])

    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')

        # Disable internode and low-latency kernels
        assert disable_nvshmem
    else:
        # Prefer H800 series
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')

        # CUDA 12 flags
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
        nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Debug build flags
    debug_env = os.getenv('DEEPEP_DEBUG', os.getenv('DEBUG', '0'))
    if str(debug_env).lower() in ('1', 'true', 'yes', 'on'):
        print('Debug build enabled: adding host/device debug flags and disabling optimizations')
        cxx_flags = [flag for flag in cxx_flags if flag != '-O3']
        nvcc_flags = [flag for flag in nvcc_flags if flag != '-O3']
        cxx_flags.extend(['-g', '-O0', '-fno-omit-frame-pointer'])
        nvcc_flags.extend(['-G', '-g', '-lineinfo', '-Xcompiler', '-g', '-Xcompiler', '-fno-omit-frame-pointer'])

    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f' > Backend: {"NIXL" if use_nixl else ("NVSHMEM" if not disable_nvshmem else "intranode-only")}')
    print()

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(name='deep_ep',
                     version='1.2.1' + revision,
                     packages=setuptools.find_packages(include=['deep_ep']),
                     ext_modules=[
                         CUDAExtension(name='deep_ep_cpp',
                                       include_dirs=include_dirs,
                                       library_dirs=library_dirs,
                                       libraries=libraries,
                                       sources=sources,
                                       extra_compile_args=extra_compile_args,
                                       extra_link_args=extra_link_args)
                     ],
                     cmdclass={'build_ext': BuildExtension})
