"""Structural analysis: canonical document → section tree, knowledge objects and text units.

Works on any :class:`~ips_r.data.models.Document`:

- *structured* documents (PDFs) are walked element by element: headings build the
  section hierarchy (numbering, level, parent, path, appendix flag); paragraphs,
  abstracts, captions, tables, equations, references and footnotes become knowledge
  objects; captions are paired with the table they describe;
- *plain* documents (QA passages) get a root section whose paragraphs are split on
  blank lines.

In-text citations (numeric ``[1, 3-5]`` and author-year ``(Smith et al., 2020)``) and
cross references (``Figure 2``, ``Table 1``, ``Section 3.1``, ``Eq. (4)``,
``Appendix A``) are detected in running text and resolved against the document's
references, figures, tables, sections and equations.

All text is normalised by a :class:`~ips_r.knowledge.normalization.TextNormalizer`
whose offset map yields exact original-document :class:`SourceSpan` objects.
:class:`TextUnit` objects are the inputs of chunk processing.

Modules: :mod:`.analyzer` (document walk), :mod:`.builders` (one builder per object
kind), :mod:`.mentions` (citations and cross references), :mod:`.patterns` (fixed
patterns, :func:`first_author`) and :mod:`.results` (outputs and per-document state).
"""

from ips_r.knowledge.structure.analyzer import StructuralAnalyzer
from ips_r.knowledge.structure.patterns import first_author
from ips_r.knowledge.structure.results import FLOW_SEPARATOR, StructureResult, TextUnit

__all__ = ["FLOW_SEPARATOR", "StructuralAnalyzer", "StructureResult", "TextUnit", "first_author"]
