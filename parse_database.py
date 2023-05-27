import json
import os
from tvm import meta_schedule as ms

from tvm.tir.tensor_intrin.arm_cpu import ARM_DOT_4x4_i8_SDOT_INTRIN, ARM_DOT_4x4_u8_UDOT_INTRIN, ARM_DOT_4x4_i8_NEON_INTRIN
from tvm import te, tir, relay

def parse_db():
    work_dir = "/Users/admin/workspace/ms_filter_tvm/tvm/mobilenet_v1/full_no_filter"

    tuning_records_path = os.path.join(work_dir, "database_tuning_record.json")

    tuning_records_json = []

    with open(tuning_records_path, 'r') as f:
        for line in f:
            line_data = json.loads(line.strip())
            tuning_records_json.append(line_data)

    sec_list = []
    sorted_tuning_records = []
    data = []

    for r in tuning_records_json:
        time = r[1][1][0]
        if time > 999:
            continue
        sec_list.append(time)
        data.append([time, r])

    sorted_tuning_records = sorted(data, key=lambda x: x[0])

    res = []
    for v in sorted_tuning_records:
        t = v[0]
        r = v[1]
        meta_data = r[1][0][1]
        if meta_data == []:
            continue
        full = []
        #check = (3,5,7,9,11,13)
        for i in meta_data:
            idx, args = i
            # if idx == check[0]:
            #     continue
            #if idx not in check:
                #continue
            full.extend(args)
        print('[{0:.20f},{1}],'.format(t, full))
        res.append([t,full])
    
    return res

parse_db()