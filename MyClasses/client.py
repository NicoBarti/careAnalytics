import json
import socket
import subprocess
import time

class Client:
    def __init__(self, PORT=8383, HOST='localhost', bufer_size=8000, printInfo=False, ComputeErrors=1,
                 ENGINE_PATH = "./engine/CareEngine5_socket.jar"):
        self.PORT = PORT
        self.HOST = HOST
        self.bufer_size = bufer_size
        self.printInfo = printInfo
        self.ComputeErrors = ComputeErrors
        self.ENGINE_PATH = ENGINE_PATH
        self.received_params = ""

    def start_server(self):
        arguments = []
        arguments.append('java')
        arguments.append('-jar')
        arguments.append(self.ENGINE_PATH)
        arguments.append(str(self.PORT))
        subprocess.Popen(arguments)
        delay_for_socket = 0.6
        time.sleep(delay_for_socket)
        return None

    def get_model_name(self):
        return self.ENGINE_PATH.split(sep='/')[-1].split(sep='.')[0]

    def build_fetchMessage(self, fetchType):
        if fetchType not in ['OK_params', 'add_progression']:
            print("fetchType not valid")
            raise Exception("fetchType not valid")
        return bytes(fetchType + '\n', 'UTF-8')

    def socket_with_model_paramGrid_2(self, gridParameters, PORT=8383, HOST='localhost', bufer_size=8000, printInfo=False,
                                      ComputeErrors=1, fetchType='OK_params'):
        """Connect to the Java ABM and return its output.

        Important paramenters:
        gridParameters - a dataframe with parameter per row
        Output:
        a dictionary with rows of grid in key and results as values
        """
        fetchMessage = self.build_fetchMessage(fetchType)
        errors = {}
        for i in range(0, ComputeErrors):
            errors[i] = {}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((str(self.HOST), int(self.PORT)))
            for err in range(0, ComputeErrors):
                data = {}
                for i in range(len(gridParameters)):
                    params = self.unpackGrid(gridParameters.iloc[i])
                    params_to_bytes = bytes(json.dumps(params) + '\n', 'UTF-8')
                    s.send(params_to_bytes)
                    self.received_params = s.recv(bufer_size)
                    if self.received_params.strip() != b'Done': ##skip param verification in this case, json.loads fails
                        if (not self.comparaParams(enviados=params, recividos=json.loads(bytes(self.received_params.strip())))):
                            s.close()
                            return ()
                    s.send(fetchMessage)
                    data_size = s.recv(bufer_size)
                    s.send(b"chunk\n")
                    results = bytearray()
                    print(self.received_params)
                    while len(results) < int(data_size):
                        results.extend(s.recv(bufer_size).strip())
                    if len(results.strip()) == int(data_size):
                        errors[err][i] = json.loads(bytes(results))
                        # data[i] = json.loads(bytes(results))
                    else:
                        print("ERROR. Expected", int(data_size), "bytes, but received ", len(results.strip()))
                        return ("")
                    s.send(b'NextCall\n')
                    afterNextCall = s.recv(bufer_size)
                    #print(afterNextCall, "after sending NextCall to server")
                    if afterNextCall == b'\n':
                        s.recv(bufer_size) ##trying to catch case where there was just one more \n bit to receive
                # errors[err] = data
            s.close()
        if (printInfo):
            print({'sent_params': params, 'PORT': PORT, 'HOST': HOST, 'buffer_size': bufer_size})
        if (ComputeErrors == 1):
            return (errors[0])
        else:
            return (errors)


    def unpackGrid(self, row):
        d = {}
        for par_name in (row).index:
            if par_name in ['capacity', 'weeks', 'numPatients', 'N', 'W', 'varsigma', 'OBS_PERIOD']:  ##handle integer-parameters
                d[par_name] = [int(row[par_name])]
            elif par_name in ['DISEASE_SEVERITY', 'LEARNING_RATE', 'SUBJECTIVE_INITIATIVE', 'SEVERITY_ALLOCATION']:  ##handle doubles
                d[par_name] = [float(row[par_name])]
            elif par_name in ['PROVIDER_INIT','PATIENT_INIT']:
                d[par_name] = [(row[par_name])]
            else:
                print(f"Unknown type {par_name}. FIX: put type in unpackGrid method in client.py")
                raise Exception(f"Unknown type {par_name}. FIX: add type in unpackGrid method in client.py")
        return (d)


    def comparaParams(self, enviados, recividos):
        for key in enviados:
            if (key == "PROVIDER_INIT" or key == "PATIENT_INIT"):
                continue
            elif (str(float(enviados[key][0])) != str(float(recividos[key]))):
                print("Parametros enviados no coinciden con los recibidos")
                print("Enviado", key, enviados[key][0])
                print("Recivido", key, recividos[key])
                raise Exception("Parametros enviados no coinciden con los recibidos")
                return (False)
        return (True)
