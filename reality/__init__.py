# reality/__init__.py
#
# Publieke interface van de reality module.
#
# De reality module is verantwoordelijk voor het samplen van werkelijke
# rijtijden op basis van empirische verdelingen uit historische data.
#
# Gebruik:
#   from reality import sample_running_time
 
from reality.sampling import sample_running_time
 
__all__ = ["sample_running_time"]
 