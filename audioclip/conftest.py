import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest

import audioclip


@pytest.fixture(autouse=True)
def _bind_public_api(request):
    """Expose the public class under the bare name used by the shipped tests.

    The test module only does ``import audioclip`` yet its placeholder smoke
    cases reference ``SoftClip`` by an unqualified name. Bind the public
    symbol onto the collected module's namespace so those cases resolve,
    without modifying the test module itself.
    """
    module = getattr(request, "module", None)
    if module is not None and not hasattr(module, "SoftClip"):
        module.SoftClip = audioclip.SoftClip
    yield
