REAL_STATION_CLUSTERS = [
    ("Columbus Circle / Union Station", 38.8972, -77.0064, "workday"),  # transit hub
    ("Lincoln Memorial", 38.8894, -77.0500, "weekend"),  # tourist/leisure
    ("Dupont Circle", 38.9097, -77.0434, "mixed"),  # residential+offices
    (
        "Eastern Market Metro / Penn Ave",
        38.8855,
        -77.0084,
        "workday",
    ),  # commuter origin
    ("Georgetown / 34th & M St NW", 38.9054, -77.0723, "weekend"),  # leisure
    ("17th & K St NW", 38.9017, -77.0395, "workday"),  # office district
    ("New York Ave & 15th St NW", 38.9010, -77.0335, "workday"),  # office/transit
    ("Jefferson Memorial", 38.8814, -77.0350, "weekend"),  # tourist
    ("Columbia Heights Metro", 38.9284, -77.0322, "mixed"),  # residential
    ("Navy Yard / Ballpark", 38.8762, -77.0042, "mixed"),  # mixed-use
]

# Solver parameters
COST_PER_KM = 1.5
PENALTY_PER_BIKE = 7.0
MIN_STATION_CAPACITY = 2

INITIAL_INVENTORY_RATIO = 0.75
TARGET_UTILIZATION_RATIO = 0.75
CAPACITY_PERCENTILE = 95
CAPACITY_STDDEV_FACTOR = 2.0

# Instance configurations (n_time, n_stations, size_label)
INSTANCE_SLICES_TEMPLATE = [
    (24, 3, "small"),
    (48, 6, "medium"),
    (120, 10, "large"),
    (None, 10, "real"),  # None means full dataset length
]
N_INSTANCES = len(INSTANCE_SLICES_TEMPLATE)
