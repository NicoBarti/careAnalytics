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
        self.received_configured_params = ""

    def start_server(self):
        arguments = []
        arguments.append('java')
        arguments.append('-jar')
        arguments.append(self.ENGINE_PATH)
        arguments.append(str(self.PORT))
        subprocess.Popen(arguments)
        delay_for_socket = 0.7
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
            #try:
            s.connect((str(self.HOST), int(self.PORT)))
            #except ConnectionRefusedError as e:
            #    print(f"(Python Client) Parece que Java refused: {e}")
            #    print(f"Tratando de nuevo")
            #    time.sleep(12)
            for err in range(0, ComputeErrors):
                data = {}
                for i in range(len(gridParameters)):
                    #params = self.unpackGrid(gridParameters.iloc[i])
                    params = self.unpack2(gridParameters.iloc[i]) #This method passess strings
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
                    print(f'from engine {self.received_params}')
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


    def unpack2(self, row):
        """Convert serie to dicitonaty with values in brakets"""
        d = {}
        for par_name in (row).index:
            d[par_name] = [str(row[par_name])]
        return d


    def unpackGrid(self, row):
        """Convert the serie"""
        d = {}
        for par_name in (row).index:
            if par_name in ['capacity', 'weeks', 'numPatients', 'N', 'W', 'varsigma', 'OBS_PERIOD', 'seed', 'totalCapacity']:  ##handle integer-parameters
                d[par_name] = [int(row[par_name])]
            elif par_name in ['DISEASE_SEVERITY', 'LEARNING_RATE', 'SUBJECTIVE_INITIATIVE', 'SEVERITY_ALLOCATION',
                              "fixed_delta", "fixed_capN", "fixed_rho", "fixed_eta", "fixed_kappa", "fixed_capE",
                              "fixed_psi", "fixed_lambda", "fixed_tau"]:  ##handle doubles
                d[par_name] = [float(row[par_name])]
            elif par_name in ['PROVIDER_INIT','PATIENT_INIT','pathfinder', 'obsH','obsN', 'obsC', 'obsT', 'obsE', 'obsB',
                              "obsSimpleC", "obsSimpleE", "obsSimpleB", "reproduce_line", "obsDisease", "obsExpNoise",
                              "obsInstExp", "obsDelta", "obsPerformance", 'obsMaxExp']: ##handle boolean
                d[par_name] = [(row[par_name])]
            elif par_name in ['Pi']: ##handle  strings
                d[par_name] = [(row[par_name])]

            else:
                print(f"Unknown type {par_name}. FIX: put type in unpackGrid method in client.py")
                raise Exception(f"Unknown type {par_name}. FIX: add type in unpackGrid method in client.py")
        return (d)


    def comparaParams(self, enviados, recividos):
        for key in enviados:
            if (key == "PROVIDER_INIT" or key == "PATIENT_INIT" or key == "pathfinder" or key == "obsH" or
                    key == "obsN" or key == "obsC" or key == "obsT" or key == "obsE" or key == "obsB"
            or key == "obsSimpleB" or key == "obsSimpleC" or key == "obsSimpleE" or key == "reproduce_line"
            or key == "obsDisease" or key == "obsExpNoise" or key == 'obsInstExp' or key == 'obsDelta'
            or key == "obsPerformance" or key == 'obsMaxExp'):
                if(str(enviados[key][0])).lower() != str(recividos[key]).lower():
                    print("Parametros enviados no coinciden con los recibidos")
                    print("Enviado", key, enviados[key][0])
                    print("Recivido", key, recividos[key])
                    raise Exception("Parametros enviados no coinciden con los recibidos")
            elif(key == "initial_h"): #Don't check initialization
                None
            elif (key == "Pi"):
                if (enviados[key][0] != str(recividos[key])):
                    print("Parametros enviados no coinciden con los recibidos")
                    print("Enviado", key, enviados[key][0])
                    print("Recivido", key, recividos[key])
                    raise Exception("Parametros enviados no coinciden con los recibidos")
            elif (str(float(recividos[key]))[0:15] != str(float(enviados[key][0]))[0:15]):
                print("Parametros enviados no coinciden con los recibidos")
                print("Enviado", key, enviados[key][0])
                print("Recivido", key, recividos[key])
                raise Exception("Parametros enviados no coinciden con los recibidos")
                return (False)

        return (True)
