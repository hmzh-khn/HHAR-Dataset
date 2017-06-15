RAW_DATA_FILENAMES = ['Phones_accelerometer', 'Phones_gyroscope', 'Watch_accelerometer', 'Watch_gyroscope']
FILENAMES = {
  'pa': RAW_DATA_FILENAMES[0],
  'pg': RAW_DATA_FILENAMES[1],
  'wa': RAW_DATA_FILENAMES[2],
  'wg': RAW_DATA_FILENAMES[3],
}

ACTIONS = ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown'] # and 'null'
ACTION_KEYS = {}
i = 0
for a in ACTIONS:
  ACTION_KEYS[a] = i
  i += 1

USERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
DEVICES = ['nexus4_1',
  'nexus4_2',
  's3_1',
  's3_2',
  's3mini_1',
  's3mini_2',
  'samsungold_1',
  'samsungold_2']
DEVICES_FREQS = {
  'nexus4_1': 200,
  'nexus4_2': 200,
  's3_1': 150,
  's3_2': 150,
  's3mini_1': 100,
  's3mini_2': 100,
  'samsungold_1': 50,
  'samsungold_2': 50,
}