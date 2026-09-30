"""The submitted manuscript (paper/submitted) prints only generated numbers, and the ones the generator produces.

The manuscript source is not distributed; these checks run only where a local copy sits in paper/submitted."""
import re

import pytest

from mtkaudit.paths import ROOT

GEN = ROOT / "paper" / "latex" / "generated" / "numbers.tex"
SUB = ROOT / "paper" / "submitted"
pytestmark = pytest.mark.skipif(not (SUB / "generated" / "numbers.tex").exists() or not GEN.exists(),
                                reason="the manuscript source is not distributed")


def macros(path):
    return dict(re.findall(r"\\newcommand\{\\n(\w+)\}\{([^}]*)\}", path.read_text()))


def test_submitted_numbers_equal_the_generated_ones():
    assert macros(SUB / "generated" / "numbers.tex") == macros(GEN)


def test_every_number_the_manuscript_uses_is_generated():
    defined = macros(SUB / "generated" / "numbers.tex")
    used = set()
    for tex in SUB.rglob("*.tex"):
        used |= set(re.findall(r"\\n([A-Z][A-Za-z]+)", tex.read_text()))
    assert not used - set(defined), sorted(used - set(defined))
