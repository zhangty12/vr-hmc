import numpy
from xlrd import open_workbook

def read_concrete():
    wb = open_workbook('concrete/Concrete_Data.xls')
    data = []
    ans = []
    for s in wb.sheets():
        for row in range(1, s.nrows):
            col_value = []
            for col in range(s.ncols-1):
                value  = s.cell(row, col).value
                col_value.append(float(value))
            data.append(col_value)
            ans.append(s.cell(row, s.ncols-1).value)

    X = numpy.array(data)
    y = numpy.array(ans)
    X_normed = X / X.max(axis = 0)
    return len(data[0]), X_normed, y
