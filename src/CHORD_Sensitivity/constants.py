import numpy as np

CHORD = {
   'name': "CHORD",
   'ndish_ew': 22,
   'ndish_ns': 24,
   'dish_separation_ew': 6.3, # m
   'dish_separation_ns': 8.5, # m
   'dish_diameter': 6.0, # m
   'efficiency': 0.5, # aperture efficiency
   'latitude': 49.320750,
   'frequencyMin': 300, # MHz
   'frequencyMax': 1500, # MHz
   'nchannels': 6000,
   'Tsys': 30.0, # K
   'ndumps': 720,
   "min_dec": 20.0, # degrees
   "max_dec": 80.0 # degrees
}

PATHFINDER = {
    'name': "PATHFINDER",
    'ndish_ew': 11,
    'ndish_ns': 6,
    'dish_separation_ew': 6.3, # m
    'dish_separation_ns': 8.5, # m
    'dish_diameter': 6.0, # m
    'efficiency': 0.5, # aperture efficiency
    'latitude': 49.320750,
    'frequencyMin': 300, # MHz
    'frequencyMax': 1500, # MHz
    'nchannels': 6000,
    'Tsys': 30.0, # K
    'ndumps': 720,
    "min_dec": 20.0, # degrees
    "max_dec": 80.0 # degrees
}

## Physical Constants
K_B = 1.38e3 # Boltzmann constant in Jy K^(-1) m^2
OMEGA = 7.29e-5  # Earth rotation rate in rad/s
C = 3e8  # speed of light in m/s

CLEAN_LINES = np.array([
    413.3882849, 444.4447839, 449.9204591, 472.7452917, 478.6917387,
    484.7383357, 516.5495982, 523.2435042, 530.0535724, 536.9823337,
    544.0323853, 551.2063928, 565.9372938, 573.4998805, 581.1978145,
    589.0341371, 605.1345288, 715.1902178, 736.070547, 746.8160182,
    791.9520966, 840.7999512, 880.1042058, 893.7491117, 921.8969082,
    966.3781549, 981.84006, 1013.767665, 1030.251531, 1081.898736,
    1175.927428, 1196.028536, 1216.590408, 1237.626322, 1259.150019,
    1281.175719, 1303.718144, 1326.792537, 1350.414681, 1374.600927,
    1399.368217, 1424.734102, 1450.716777, 1477.335105
])*1e6  # convert to Hz
