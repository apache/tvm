import json

from tvm import meta_schedule as ms

from tvm.tir.tensor_intrin.arm_cpu import ARM_DOT_4x4_i8_SDOT_INTRIN, ARM_DOT_4x4_u8_UDOT_INTRIN, ARM_DOT_4x4_i8_NEON_INTRIN
from tvm import te, tir, relay

work_dir = "my_dir/"

tuning_records_path = work_dir + "database_tuning_record.json"

tuning_records_json = []

print("Starting to process records as json.\n Loading json records...")
with open(tuning_records_path, 'r') as f:
    for line in f:
        line_data = json.loads(line.strip())
        tuning_records_json.append(line_data)

sec_list = []
sorted_tuning_records = []
data = []

print("Sorting records in ascending order by 'run secs'...")
for r in tuning_records_json:
    time = r[1][1][0]
    sec_list.append(time)
    data.append([time, r])

sorted_tuning_records = sorted(data, key=lambda x: x[0])

for v in sorted_tuning_records:
    t = v[0]
    r = v[1]
    meta_data = r[1][0][1]
    full = []
    # check = (3,5,7,9,11,13)
    for i in meta_data:
        idx, args = i
        # assert idx in check
        full.extend(args)
    print('{0:.20f}'.format(t), "|".join(["{0:2}".format(i) for i in full]))
    # print('{0:.20f}'.format(t), full[7])
    
