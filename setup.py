"""Build semita's C extension on the stable ABI, with the compiler's warnings on."""

import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

SOURCES = ["csrc/module.c", "csrc/jps.c", "csrc/dijkstra.c"]
HEADERS = ["csrc/search.h", "csrc/heap.h"]


class BuildWithWarnings(build_ext):
    """Compile with warnings on, and with warnings as errors where `SEMITA_WARNINGS_AS_ERRORS` is 1, as on CI."""

    def build_extensions(self) -> None:
        as_errors = os.environ.get("SEMITA_WARNINGS_AS_ERRORS") == "1"
        if self.compiler.compiler_type == "msvc":
            flags = ["/W4", *(["/WX"] if as_errors else [])]
        else:
            flags = ["-std=c11", "-Wall", "-Wextra", "-Wconversion", *(["-Werror"] if as_errors else [])]
        for extension in self.extensions:
            extension.extra_compile_args = flags
        super().build_extensions()


setup(
    ext_modules=[
        Extension(
            "semita._search",
            sources=SOURCES,
            depends=HEADERS,
            include_dirs=["csrc"],
            define_macros=[("Py_LIMITED_API", "0x030C0000")],
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": BuildWithWarnings},
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
