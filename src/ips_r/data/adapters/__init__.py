"""Dataset adapters. Importing this package registers every adapter in ``DATASETS``."""

from ips_r.data.adapters.academic_pdf import AcademicPDFDataset
from ips_r.data.adapters.hotpotqa import HotpotQADataset
from ips_r.data.adapters.ms_marco import MSMarcoDataset
from ips_r.data.adapters.natural_questions import NaturalQuestionsDataset
from ips_r.data.adapters.squad import SquadDataset

__all__ = [
    "AcademicPDFDataset",
    "HotpotQADataset",
    "MSMarcoDataset",
    "NaturalQuestionsDataset",
    "SquadDataset",
]
