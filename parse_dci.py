import xcalparser
import pandas as pd
import code
import random

tab = xcalparser.TableTypeMsg()

result = []
n = 1000
while n > 0:
    if len(tab.data) > 0:
      result.append(tab.data[-1])
    n -= 1
    tab.parse_next()

df = pd.concat(result, ignore_index=True)
df.to_csv("./DCI-{}.csv", random.randint(0, 10))
# code.interact(local=locals())
