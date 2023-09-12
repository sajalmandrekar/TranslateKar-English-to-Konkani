print("Loading the libraries, this may take some time...")

import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np
import os
import time


class TranslateKarModel:

    def __init__(self,MODEL_PATH,en_to_kok=True):

        print("Loading the model, this may take some time...")

        #delay for 2 second
        time.sleep(2)

        self.__model = tf.saved_model.load(MODEL_PATH)

        #warming up
        print('warming up...')

        if en_to_kok == True:
            test_input='how are you?'
        else:
            test_input='तूं कसो आसा?'

        test_output = self.__model(tf.constant(test_input)) #warmup
        print("testing...ignore this:",test_output.numpy().decode())
        del test_output

        print(f'Model {MODEL_PATH} loaded!')

    def translate_text(self, input_text):
        return self.__model(tf.constant(input_text)).numpy().decode()

if __name__ == '__main__':

    eng_kok_model_path=os.path.abspath(r'./saved_models/eng-kok/translator')
    kok_eng_model_path=os.path.abspath(r'./saved_models/kok-eng/translator')

    print("\n----- TranslateKar: English to Konkani & vice versa language translator -----\n")


    while True:
        print("----- Menu -----")
        print("1. English to Konkani (default)")
        print("2. Konkani to English (default)")
        print("3. Specify model path")
        print("\nEnter any of the above options (enter q to quit):")
        choice = input()
        print()

        if choice in ('q','Q'):
            print("exiting program...")
            break
        elif choice == '1':
            default_model_path=eng_kok_model_path
        elif choice == '2':
            default_model_path=kok_eng_model_path
        elif choice == '3':
            input_path = input("Enter path: ")
            if os.path.isdir(input_path):
                default_model_path=os.path.abspath(input_path)
            else:
                print("Model file not found at specified location!")
                continue
        else:
            print("Entered incorrect option! please retry!")
            continue

        print(f"Model located at {default_model_path}\n")
        model = TranslateKarModel(default_model_path)

        while True:
            input_text = input("Enter text (enter /q to quit prompt): ")
            if input_text in ('/q','/Q','/quit'):
                break
            output_text = model.translate_text(input_text)
            print("Translated Text:",output_text)
            print()
