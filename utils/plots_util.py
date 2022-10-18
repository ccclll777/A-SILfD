
def float_to_int(data,normalization = False,normalization_limit = 100):
    if  isinstance(data[0],list):
        for i in range(len(data)):
            for j in range(len(data[i])):
                if normalization:
                    data[i][j] = int(data[i][j]/normalization_limit)
                else:
                    data[i][j] = int(data[i][j])
    else:
        for i in range(len(data)):
            if normalization:
                data[i] = int(data[i] / normalization_limit)
            else:
                data[i] = int(data[i])
    return data
def smooth(data, sm=0.8):
    smooth_data = []
    if isinstance(data[0], list):
        for d in data:
            last = d[0]
            smoothed = []
            for point in d:
                smoothed_val = last * sm + (1 - sm) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            smooth_data.append(smoothed)
        return smooth_data
    else:
        last = data[0]
        smoothed = []
        for point in data:
            smoothed_val = last * sm + (1 - sm) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
