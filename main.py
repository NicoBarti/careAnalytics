from scripts.abstract import *
from scripts.noHetero.CAPACITY_DISEASE_specialPairPlots import *
from scripts.noHetero.DISEASE_SUBJECTIVE_specialPairPlots import *
from scripts.ten_Diseases.n_visitHistograms import *
from scripts.abstract2 import do_abstract2

ENGINE_PATH="./engine/CareEngine5_socket_noHetero.jar"
#ENGINE_PATH="./engine/CareEngine5_socket_10Diseases.jar"
if __name__ == '__main__':
    print('starting')
    # routines for 10 Diseases
    #c = Client(ENGINE_PATH)
    #c.start_server()
    #plot_histograms_visits(c)

    # routines for noHetero
    #c = Client(ENGINE_PATH="./engine/CareEngine5_socket_noHetero.jar")
    #c.start_server()
    #do_expectations_converge(c)
    #do_disease_subjective_special_pairs(ENGINE_PATH="./engine/CareEngine5_socket_noHetero.jar")
    #do_capacity_DISEASE_special_pairs(ENGINE_PATH="./engine/CareEngine5_socket_noHetero.jar")
    #do_capacity_DISEASE_triptics(ENGINE_PATH="./engine/CareEngine5_socket_noHetero.jar")
    #plotVarianceSubjective(c)

    #do_subjective_capacity_special_pairs(c)
    # do_pairs(c = c,all_pairs = {
    # 0: [{'LEARNING_RATE': [0,0.5,1,1.5,2,2.5,3,4,5]},
    #     'DISEASE_SEVERITY', 'pairs_capacity_disease_1000patients'],

    # 0: ['capacity', 'DISEASE_SEVERITY', 'pairs_capacity_disease_1000patients'],
    # 1: ['capacity', 'LEARNING_RATE', 'pairs_capacity_learning_1000patients'],
    # 2: ['capacity', 'SUBJECTIVE_INITIATIVE', 'pairs_capacity_subjective_1000patients'],
    # 3: ['DISEASE_SEVERITY', 'LEARNING_RATE', 'pairs_disease_learning_1000patients'],
    # 4: ['DISEASE_SEVERITY', 'SUBJECTIVE_INITIATIVE', 'pairs_disease_subjective_1000patients'],
    # 5: ['LEARNING_RATE', 'SUBJECTIVE_INITIATIVE', 'pairs_learning_subjective_1000patients'],
    # 6: ['SUBJECTIVE_INITIATIVE', 'capacity', 'pairs_subjective_capacity_1000patients' ]
#    })


    ## ABSTRACT
    #do_abstract()
    do_abstract2()



