from MyClasses.client import Client
import pandas as pd

ENGINE_PATH="../engine/CareEngine6.jar"

c = Client(ENGINE_PATH=ENGINE_PATH)
c.start_server()
params = {"N": [1000], "W": [1], "varsigma": [150]}
data = c.socket_with_model_paramGrid_2(pd.DataFrame.from_dict(params))
print(data)

