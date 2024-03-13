import os
import shutil
import random

def split_vectors(source_folder, folder_1, folder_2,source_folder2, folder_1a, folder_2a, num_vectors_folder_1, num_vectors_folder_2):
    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return
    
    # Create destination folders if they don't exist
    os.makedirs(folder_1, exist_ok=True)
    os.makedirs(folder_2, exist_ok=True)
    
    # List all files in the source folder
    all_files1 = os.listdir(source_folder)
    all_files2 = os.listdir(source_folder2)
    
    # Open the file in read mode
    with open('/TESIS/DATOS_TESIS2/rand1.txt', 'r') as file:
        # Read all lines and convert them to float
        random_list_1 = [int(line.strip()) for line in file]

    # Open the file in read mode
    with open('/TESIS/DATOS_TESIS2/rand2.txt', 'r') as file:
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

    # # Specify the size of the second list
    # size_list_2 = 2500

    # # Randomly select numbers to fill the second list
    # random_list_2 = random.sample(random_list_2, size_list_2)

    # Print or use the generated lists as needed
    # print("Random List 1:", random_list_1)
    # print("Random List 2:", random_list_2)
    
    # Select vectors for each folder
    # vectors_folder_1 = all_files[:num_vectors_folder_1]
    # vectors_folder_2 = all_files[num_vectors_folder_1:num_vectors_folder_1 + num_vectors_folder_2]
    vectors_folder_1=[]
    vectors_folder_2=[]
    vectors_foldera_1=[]
    vectors_foldera_2=[]
    for ind in random_list_1:
        vectors_folder_1.append(all_files1[ind-1])
        vectors_foldera_1.append(all_files2[ind-1])

    for ind in random_list_2:
        vectors_folder_2.append(all_files1[ind-1])
        vectors_foldera_2.append(all_files2[ind-1])

    
    # Copy selected vectors to destination folders
    for vector_file in vectors_folder_1:
        source_path = os.path.join(source_folder, vector_file)
        dest_path = os.path.join(folder_1, vector_file)
        shutil.copy2(source_path, dest_path)
    
    for vector_file in vectors_folder_2:
        source_path = os.path.join(source_folder, vector_file)
        dest_path = os.path.join(folder_2, vector_file)
        shutil.copy2(source_path, dest_path)

    # Copy selected vectors to destination folders
    for vector_file in vectors_foldera_1:
        source_path = os.path.join(source_folder2, vector_file)
        dest_path = os.path.join(folder_1a, vector_file)
        shutil.copy2(source_path, dest_path)
    
    for vector_file in vectors_foldera_2:
        source_path = os.path.join(source_folder2, vector_file)
        dest_path = os.path.join(folder_2a, vector_file)
        shutil.copy2(source_path, dest_path)


def main():
    # Example usage:
    source_folder = "/TESIS/DATOS_TESIS2/target_from_raw"
    folder_1 = "/TESIS/DATOS_TESIS2/onepw_train"
    folder_2 = "/TESIS/DATOS_TESIS2/onepw_test"

    source_folder2 = "/TESIS/DATOS_1/input_id"
    folder_1a = "/TESIS/DATOS_1/rf_train"
    folder_2a = "/TESIS/DATOS_1/rf_test"
    
    num_vectors_folder_1 = 10000
    num_vectors_folder_2 = 2500
    split_vectors(source_folder, folder_1, folder_2,source_folder2, folder_1a, folder_2a, num_vectors_folder_1, num_vectors_folder_2)


if __name__ == '__main__':
    main()
