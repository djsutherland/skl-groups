__version__ = '0.1.0'

try:
    # This variable is injected by setup.py to let us know that we're in
    # the build process right now.
    __SKL_GROUPS_ACCEL_SETUP__
except NameError:
    __SKL_GROUPS_ACCEL_SETUP__ = False
