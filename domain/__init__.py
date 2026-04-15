# Publieke interface van de domain module.
# Alle andere modules importeren domeinobjecten via:
#
#   from domain import Train, TrainType, TrainSubtype
#   from domain import Segment, SegmentType
#   from domain import Timetable, ScheduledTimes

from domain.train import Train, TrainType, TrainSubtype
from domain.segment import Segment, SegmentType
from domain.schedule import Timetable, ScheduledTimes

__all__ = [
    "Train",
    "TrainType",
    "TrainSubtype",
    "Segment",
    "SegmentType",
    "Timetable",
    "ScheduledTimes",
]