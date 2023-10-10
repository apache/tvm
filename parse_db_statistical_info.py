import json

from tvm import meta_schedule as ms

from tvm.tir.tensor_intrin.arm_cpu import ARM_DOT_4x4_i8_SDOT_INTRIN, ARM_DOT_4x4_u8_UDOT_INTRIN, ARM_DOT_4x4_i8_NEON_INTRIN
from tvm import te, tir, relay


parsed_db_path = "/Users/admin/workspace/ms_filter_tvm/tvm/mobilenet_v1_fp32_full_no_filter.txt"


with open(parsed_db_path, 'r') as file:
    # Read the file contents
    file_content = file.readlines()

# Initialize an empty list to store arrays
arrays = []

# Iterate over each line and create arrays
for line in file_content:
    line = line.strip()  # Remove leading/trailing whitespaces
    array = eval(line)   # Evaluate the line as a Python expression
    arrays.append(array) # Append the array to the list

unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][0])

print("#################")

for array in arrays:
    print(array[0][1])
    
    if array[0][1] in unique_val:
        unique_val[array[0][1]] += 1
    else:
        unique_val[array[0][1]] = 1
    val_time.append((array[0][1], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)
unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][2])
    
    if array[0][2] in unique_val:
        unique_val[array[0][2]] += 1
    else:
        unique_val[array[0][2]] = 1
    val_time.append((array[0][2], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)
unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][3])
    
    if array[0][3] in unique_val:
        unique_val[array[0][3]] += 1
    else:
        unique_val[array[0][3]] = 1
    val_time.append((array[0][3], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)
unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][4])
    
    if array[0][4] in unique_val:
        unique_val[array[0][4]] += 1
    else:
        unique_val[array[0][4]] = 1
    val_time.append((array[0][4], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)
unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][5])
    
    if array[0][5] in unique_val:
        unique_val[array[0][5]] += 1
    else:
        unique_val[array[0][5]] = 1
    val_time.append((array[0][5], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)
unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][6])
    
    if array[0][6] in unique_val:
        unique_val[array[0][6]] += 1
    else:
        unique_val[array[0][6]] = 1
    val_time.append((array[0][6], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)
unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][7])
    
    if array[0][7] in unique_val:
        unique_val[array[0][7]] += 1
    else:
        unique_val[array[0][7]] = 1
    val_time.append((array[0][7], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)
unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][8])
    
    if array[0][8] in unique_val:
        unique_val[array[0][8]] += 1
    else:
        unique_val[array[0][8]] = 1
    val_time.append((array[0][8], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)
unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][9])
    
    if array[0][9] in unique_val:
        unique_val[array[0][9]] += 1
    else:
        unique_val[array[0][9]] = 1
    val_time.append((array[0][9], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)
unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][10])
    
    if array[0][10] in unique_val:
        unique_val[array[0][10]] += 1
    else:
        unique_val[array[0][10]] = 1
    val_time.append((array[0][10], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)

unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][11])
    
    if array[0][11] in unique_val:
        unique_val[array[0][11]] += 1
    else:
        unique_val[array[0][11]] = 1
    val_time.append((array[0][11], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)

unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][12])
    
    if array[0][12] in unique_val:
        unique_val[array[0][12]] += 1
    else:
        unique_val[array[0][12]] = 1
    val_time.append((array[0][12], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)

unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][13])
    
    if array[0][13] in unique_val:
        unique_val[array[0][13]] += 1
    else:
        unique_val[array[0][13]] = 1
    val_time.append((array[0][13], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)

unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][14])
    
    if array[0][14] in unique_val:
        unique_val[array[0][14]] += 1
    else:
        unique_val[array[0][14]] = 1
    val_time.append((array[0][14], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)

unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][15])
    
    if array[0][15] in unique_val:
        unique_val[array[0][15]] += 1
    else:
        unique_val[array[0][15]] = 1
    val_time.append((array[0][15], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)

unique_val = {}
val_time = []
# Print the arrays
for array in arrays:
    print(array[0][16])
    
    if array[0][16] in unique_val:
        unique_val[array[0][16]] += 1
    else:
        unique_val[array[0][16]] = 1
    val_time.append((array[0][16], array[0][0]))
# Сортировка словаря по ключу


sorted_counts = dict(sorted(unique_val.items()))

# Вывод результата подсчета уникальных значений
total_count = sum(sorted_counts.values())
print("Уникальные значения:")
for value, count in sorted_counts.items():
    print(f"{value}: {count}")
    
values_dict = {}
for x, y in val_time:
    if x in values_dict:
        values_dict[x].append(y)
    else:
        values_dict[x] = [y]

# Вычисление среднего значения y для каждого x
averages = {}
for x, y_values in values_dict.items():
    avg_y = sum(y_values) / len(y_values)
    averages[x] = avg_y

# Вывод результата
for x, avg_y in sorted(averages.items()):
    print(f"{avg_y}")

print("Общее количество элементов:", total_count)
print('*'*20)