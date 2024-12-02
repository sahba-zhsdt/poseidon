from netCDF4 import Dataset

dataset = Dataset('./data/CE-RPUI.nc', 'r')
data = dataset.variables['data']
print(dataset)

print("\nDimensions:")
for dim in dataset.dimensions.values():
    print(dim)
    
print("\nVariables:")
for var in dataset.variables.values():
    print(var)   
    
print("\nGlobal Attributes:")
for attr in dataset.ncattrs():
    print(f"{attr}: {dataset.getncattr(attr)}")