import os
import shutil
import random

def split_vectors(source_folder_RF, folder_RF_train, folder_RF_test, 
                  source_folder_ENH, folder_ENH_train, folder_ENH_test,
                  source_folder_ONEPW, folder_ONEPW_train, folder_ONEPW_test):
    
    # List all files in the source folder
    all_files_RF = os.listdir(source_folder_RF)
    all_files_ENH = os.listdir(source_folder_ENH)
    all_files_ONEPW = os.listdir(source_folder_ONEPW)
    
    # Open the file in read mode
    with open('/nfs/privileged/edgar/Docs/rand1.txt', 'r') as file:
        # Read all lines and convert them to float
        random_list_1 = [int(line.strip()) for line in file]

    # Open the file in read mode
    with open('/nfs/privileged/edgar/Docs/rand2.txt', 'r') as file:
    # Read all lines and convert them to float
        random_list_2 = [int(line.strip()) for line in file]


    # # Randomly shuffle the list of files
    # random.shuffle(all_files)

    ####

    # import random

    # # Generate a list of integers from 1 to 12500
    # all_numbers = list(range(1, 12501))

    # # Specify the size of the first list
    # size_list_1 = 10000

    # # Generate the first random list
    # random_list_1 = random.sample(all_numbers, size_list_1)
    # # Specify the file path where you want to save the numbers
    # file_path1 = "/TESIS/DATOS_TESIS2/rand1.txt"

    # # Open the file in write mode and write the numbers
    # with open(file_path1, 'w') as file:
    #     for number in random_list_1:
    #         file.write(str(number) + '\n')

    # print(f"The list has been successfully written to {file_path1}.")

    # # Create the second list by excluding numbers from the first list
    # random_list_2 = [num for num in all_numbers if num not in random_list_1]
    # # Specify the file path where you want to save the numbers
    # file_path2 = "/TESIS/DATOS_TESIS2/rand2.txt"

    # # Open the file in write mode and write the numbers
    # with open(file_path2, 'w') as file:
    #     for number in random_list_2:
    #         file.write(str(number) + '\n')

    # print(f"The list has been successfully written to {file_path2}.")

    ####
        
    vectors_folder_RF_train=[]
    vectors_folder_RF_test=[]

    vectors_folder_ENH_train=[]
    vectors_folder_ENH_test=[]

    vectors_folder_ONEPW_train=[]
    vectors_folder_ONEPW_test=[]
    
    for ind in random_list_1:
        vectors_folder_RF_train.append(all_files_RF[ind-1])
        vectors_folder_ENH_train.append(all_files_ENH[ind-1])
        vectors_folder_ONEPW_train.append(all_files_ONEPW[ind-1])

    for ind in random_list_2:
        vectors_folder_RF_test.append(all_files_RF[ind-1])
        vectors_folder_ENH_test.append(all_files_ENH[ind-1])
        vectors_folder_ONEPW_test.append(all_files_ONEPW[ind-1])

    
    # Copy selected vectors to destination folders
    for vector_file in vectors_folder_RF_train:
        source_path = os.path.join(source_folder_RF, vector_file)
        dest_path = os.path.join(folder_RF_train, vector_file)
        shutil.copy2(source_path, dest_path)
    
    for vector_file in vectors_folder_RF_test:
        source_path = os.path.join(source_folder_RF, vector_file)
        dest_path = os.path.join(folder_RF_test, vector_file)
        shutil.copy2(source_path, dest_path)

    # Copy selected vectors to destination folders
    for vector_file in vectors_folder_ENH_train:
        source_path = os.path.join(source_folder_ENH, vector_file)
        dest_path = os.path.join(folder_ENH_train, vector_file)
        shutil.copy2(source_path, dest_path)
    
    for vector_file in vectors_folder_ENH_test:
        source_path = os.path.join(source_folder_ENH, vector_file)
        dest_path = os.path.join(folder_ENH_test, vector_file)
        shutil.copy2(source_path, dest_path)

    # Copy selected vectors to destination folders
    for vector_file in vectors_folder_ONEPW_train:
        source_path = os.path.join(source_folder_ONEPW, vector_file)
        dest_path = os.path.join(folder_ONEPW_train, vector_file)
        shutil.copy2(source_path, dest_path)
    
    for vector_file in vectors_folder_ONEPW_test:
        source_path = os.path.join(source_folder_ONEPW, vector_file)
        dest_path = os.path.join(folder_ONEPW_test, vector_file)
        shutil.copy2(source_path, dest_path)


def main():

    source_folder_RF = "/nfs/privileged/isalazar/datasets/simulatedCystDataset/TUFFC/input_id"
    folder_RF_train = "/nfs/privileged/edgar/datasets/dataRF/RF_train"
    folder_RF_test = "/nfs/privileged/edgar/datasets/dataRF/RF_test"

    source_folder_ENH = "/nfs/privileged/isalazar/datasets/simulatedCystDataset/TUFFC/target_enh"
    folder_ENH_train = "/nfs/privileged/edgar/datasets/dataENH/ENH_train"
    folder_ENH_test = "/nfs/privileged/edgar/datasets/dataENH/ENH_test"

    source_folder_ONEPW = "/nfs/privileged/isalazar/datasets/simulatedCystDataset/TUFFC/target_from_raw"
    folder_ONEPW_train = "/nfs/privileged/edgar/datasets/dataONEPW/ONEPW_train"
    folder_ONEPW_test = "/nfs/privileged/edgar/datasets/dataONEPW/ONEPW_test"
    
    split_vectors(source_folder_RF, folder_RF_train, folder_RF_test,
                  source_folder_ENH, folder_ENH_train, folder_ENH_test,
                  source_folder_ONEPW, folder_ONEPW_train, folder_ONEPW_test)


if __name__ == '__main__':
    main()
