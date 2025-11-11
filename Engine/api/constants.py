import logging

LOG_LEVEL = logging.DEBUG

# record uploaded files
RECORD_UPLOAD = True

# Time Formatting
MAX_MINUTES_FLOW = 300  # the maximum flow time before coverting to hours
MAX_MINUTES_PROCESS = 120  # maximum number of minutes before converting process times to hours

# Printing
SENS_ROUND_DEC = 5
TIME_ROUND_DEC = 3
COST_ROUND_DEC = 2
OTHER_ROUND_DEC = 3
TAT_ROUND_DAYS = 1
SMART_SIGFIGS = 3

# Epsilon
epsilon = 1.0e-10

# Error handling
THROW_HTTP_ERROR = True

# User Type logins
USE_USERTYPE_LOGIN = False
