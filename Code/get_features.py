import pandas as pd
import os
from tqdm import tqdm
import argparse
import re


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Inception_v3 model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=".\Data",
        help="Path to the data folder",
        required=True,
    )
    args = parser.parse_args()
    return args


def main():
    patient_ID_list = []
    Label_list = []
    args = parse_arguments()
    for (root, dirs, files) in os.walk(args.data_path, topdown=True):
        for file in files:
            if file.endswith((".wav", ".WAV")):
                path = os.path.join(root, file)
                print(f"processing file {path}")

                os.system(
                    f".\opensmile-3.0.1-win-x64\\bin\SMILExtract.exe -C ..\config\is09-13\IS09_emotion.conf -I {path} --csvoutput temp.csv"
                )

                patient_ID_list.append(re.search("[A-Z][0-9]+", root).group(0))
                if re.search("Healthy", root):  # Health: 0 | Patient: 1
                    Label_list.append(0)
                else:
                    Label_list.append(1)

    df = pd.read_csv("temp.csv", sep=";")
    print(df.shape)
    print(len(Label_list))
    print(len(patient_ID_list))
    df["Label"] = Label_list
    df["patient_ID"] = patient_ID_list
    df.drop(["name", "frameTime"], axis=1, inplace=True)
    df.to_csv("SLI_INTERSPEECH2009_Functionals.csv", index=False)


if __name__ == "__main__":
    main()
