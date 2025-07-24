import psutil
print(f"Available RAM: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
