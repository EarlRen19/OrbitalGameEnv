import importlib.metadata as metadata
import os
import platform
import warnings

packagedir = os.path.abspath(os.path.dirname(__file__))

# Make sure to adjust the filter to show DeprecationWarning
warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)

if platform.system() == "Windows":
    try:
        import ctypes

        ctypes.CDLL("vcruntime140.dll")
        ctypes.CDLL("msvcp140.dll")
    except OSError:
        raise OSError(
            """Microsoft Visual C++ Redistribution Pack is not installed.
It can be downloaded from https://aka.ms/vs/16/release/vc_redist.x64.exe."""
        )

    # Loading DLLs on Windows is kind of a disaster
    # The best approach seems to be using LoadLibraryEx with user defined search paths.
    # This kind of acts like $ORIGIN or @loader_path on Unix / macOS.
    # This way we guarantee we load OUR DLLs.
    os.add_dll_directory(packagedir)

__version__ = metadata.version(__package__)

from oge_py._oge_py import SatState, OGESettings, OGEInterface, ma2ta, coe2rv, solar_illumination_angle

__all__ = ["SatState", "OGESettings", "OGEInterface", "ma2ta", "coe2rv", "solar_illumination_angle"]

try:
    from oge_py.env import OGEEnv, OGEEnvCfg, OGEEnvStepMetadata
    __all__ += ["OGEEnv", "OGEEnvCfg", "OGEEnvStepMetadata"]
except ImportError:
    pass

try:
    from oge_py.vector_env import OGEVectorEnv, OGEVectorEnvCfg
    from oge_py._oge_py import OGEVectorInterface
    __all__ += ["OGEVectorEnv", "OGEVectorEnvCfg", "OGEVectorInterface"]
except ImportError:
    pass
