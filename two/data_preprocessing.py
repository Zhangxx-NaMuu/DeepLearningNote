import os
import pandas as pd
import torch


def makedir_csv():
    os.makedirs(os.path.join('.', 'data'), exist_ok=True)
    data_files = os.path.join('.', 'data', 'house_tiny.csv')
    with open(data_files, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
    return data_files


def Handle_missing_values(csv_files):
    # 将连续的NAN值取平均值进行填充
    inputs, outputs = csv_files.iloc[:, 0:2], csv_files.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean())
    return inputs, outputs


def get_dummy(inputs):
    # 将离散的NAN值置为0或1
    inputs = pd.get_dummies(inputs, dummy_na=True)
    # print(inputs)
    return inputs


# ****将输入转换成张量****
def data_to_tensor(input, output):
    x, y = torch.tensor(input.values), torch.tensor(output.values)
    return x, y


data_file = makedir_csv()
data = pd.read_csv(data_file)
input, output = Handle_missing_values(data)
inputs = get_dummy(input)
x, y = data_to_tensor(inputs, output)
# print(x)
# print(y)

# print(inputs)
# print(data)

