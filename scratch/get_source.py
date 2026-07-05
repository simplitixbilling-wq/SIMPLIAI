import inspect
from app_core.bridge import Bridge

b = Bridge()
source = inspect.getsource(b._generate)
print(source)
