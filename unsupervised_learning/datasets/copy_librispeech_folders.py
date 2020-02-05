import os

librispeech_dir = "librispeech_train_clean_100_wav"


count = 0

for folderA in os.listdir(librispeech_dir):
    folderAPath = os.path.join(librispeech_dir, folderA)

    for folderB in os.listdir(folderAPath):
        folderBPath = os.path.join(folderAPath, folderB)

        for file in os.listdir(folderBPath):
            os.rename(os.path.join(folderBPath, file), os.path.join(folderAPath, file))
            count += 1

        os.removedirs(folderBPath)
        print("processed: ", count)
