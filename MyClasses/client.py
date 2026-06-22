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
                                      ComputeErrors=1, fetchType='OK_params', batch_size=100):
        """Connect to the Java ABM and return its output.

        Important paramenters:
        gridParameters - a dataframe with parameter per row
        Output:
        a dictionary with rows of grid in key and results as values
        """
        simulation_map = []
        batch_params = []
        for err in range(ComputeErrors):
            for i in range(len(gridParameters)):
                params = self.unpack2(gridParameters.iloc[i])
                simulation_map.append((err, i))
                batch_params.append(params)
        
        errors = {}
        for err in range(ComputeErrors):
            errors[err] = {}

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((str(self.HOST), int(self.PORT)))
            with s.makefile('r', encoding='utf-8') as sock_file:
                for chunk_idx in range(0, len(batch_params), batch_size):
                    chunk_params = batch_params[chunk_idx:chunk_idx + batch_size]
                    chunk_map = simulation_map[chunk_idx:chunk_idx + batch_size]
                    
                    payload = json.dumps(chunk_params) + '\n'
                    s.sendall(payload.encode('utf-8'))
                    
                    response_line = sock_file.readline()
                    if not response_line:
                        print("ERROR: Connection closed by server or empty response.")
                        return {}
                    
                    try:
                        chunk_results = json.loads(response_line)
                    except json.JSONDecodeError as e:
                        print(f"ERROR: Failed to parse batch response JSON: {e}")
                        print(f"Raw response: {response_line[:500]}")
                        return {}
                    
                    if isinstance(chunk_results, dict) and "error" in chunk_results:
                        print(f"Server Error: {chunk_results['error']}")
                        return {}
                    
                    for idx, res in enumerate(chunk_results):
                        err, i = chunk_map[idx]
                        orig_params = chunk_params[idx]
                        
                        if "resolved_params" in res:
                            resolved = res["resolved_params"]
                            try:
                                if not self.comparaParams(enviados=orig_params, recividos=resolved):
                                    print(f"Validation failed for simulation (err={err}, idx={i})")
                                    return {}
                            except Exception as ex:
                                print(f"Validation exception for simulation (err={err}, idx={i}): {ex}")
                                return {}
                        
                        errors[err][i] = res
                        
        if printInfo:
            print({'sent_batch_size': len(batch_params), 'PORT': PORT, 'HOST': HOST, 'batch_size': batch_size})
            
        if ComputeErrors == 1:
            return errors[0]
        else:
            return errors


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
                              "fixed_psi", "fixed_lambda", "fixed_tau", "prioritization_granularity"]:  ##handle doubles
                d[par_name] = [float(row[par_name])]
            elif par_name in ['PROVIDER_INIT','PATIENT_INIT','pathfinder', 'obsH','obsN', 'obsC', 'obsT', 'obsE', 'obsB',
                              "obsSimpleC", "obsSimpleE", "obsSimpleB", "reproduce_line", "obsDisease", "obsExpNoise",
                              "obsInstExp", "obsDelta", "obsPerformance", 'obsMaxExp', 'stepPerformance']: ##handle boolean
                d[par_name] = [(row[par_name])]
            elif par_name in ['Pi']: ##handle  strings
                d[par_name] = [(row[par_name])]

            else:
                print(f"Unknown type {par_name}. FIX: put type in unpackGrid method in client.py")
                raise Exception(f"Unknown type {par_name}. FIX: add type in unpackGrid method in client.py")
        return (d)


    def comparaParams(self, enviados, recividos):
        for key in enviados:
            if key not in recividos:
                continue
            if (key == "PROVIDER_INIT" or key == "PATIENT_INIT" or key == "pathfinder" or key == "obsH" or
                    key == "obsN" or key == "obsC" or key == "obsT" or key == "obsE" or key == "obsB"
            or key == "obsSimpleB" or key == "obsSimpleC" or key == "obsSimpleE" or key == "reproduce_line"
            or key == "obsDisease" or key == "obsExpNoise" or key == 'obsInstExp' or key == 'obsDelta'
            or key == "obsPerformance" or key == 'obsMaxExp' or key == 'stepPerformance'):
                if(str(enviados[key][0])).lower() != str(recividos[key]).lower():
                    print("Parametros enviados no coinciden con los recibidos")
                    print("Enviado", key, enviados[key][0])
                    print("Recivido", key, recividos[key])
                    raise Exception("Parametros enviados no coinciden con los recibidos")
            elif(key == "initial_h" or (key == "seed" and str(enviados[key][0]) in ["0", "0.0"])): #Don't check initialization or random seed
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
