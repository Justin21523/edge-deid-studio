import time, numpy as np
from pii_models import get_detector
det = get_detector("zh")
txt = "王小明住在台北市信義路100號，其身分證號A123456789，Email x@example.com。" * 5

for _ in range(3): det.detect(txt)  # warm-up
t0 = time.time()
for _ in range(30): det.detect(txt)
print("avg(ms):", (time.time()-t0)/30*1000)
