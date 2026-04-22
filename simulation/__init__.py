# simulation/__init__.py
#
# Publieke interface van de simulation module.
#
# Gebruik:
#   from simulation import Simulator, SystemState, EventQueue, Dispatcher
 
from simulation.state       import SystemState
from simulation.event_queue import EventQueue, TrainEntered, TrainExited
from simulation.dispatcher  import Dispatcher
from simulation.simulator   import Simulator
 
__all__ = [
    "SystemState",
    "EventQueue",
    "TrainEntered",
    "TrainExited",
    "Dispatcher",
    "Simulator",
]
 