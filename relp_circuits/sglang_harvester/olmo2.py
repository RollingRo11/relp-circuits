"""SGLang external-package entry shim for OLMo-3 capture.

SGLang's `SGLANG_EXTERNAL_MODEL_PACKAGE` resolver imports this module's
qualified name, runs its top-level code, and reads `EntryClass`. We use that
import to install the universal ForwardBatch patch and the MLP-level capture
patches on `Olmo2*` classes (Olmo3 routes through Olmo2 in SGLang 0.5.9).

Set in the driver before `import sglang`:

    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "relp_circuits.sglang_harvester.olmo2"
"""

from __future__ import annotations

import atexit

from relp_circuits.sglang_harvester._capture import (
    _install_signal_handlers,
    flush_state_to_disk,
    install_universal_patches,
    patch_olmo3_capture,
)

install_universal_patches()
patch_olmo3_capture()
_install_signal_handlers()
atexit.register(flush_state_to_disk)

from sglang.srt.models.olmo2 import Olmo2ForCausalLM  # noqa: E402

EntryClass = [Olmo2ForCausalLM]
