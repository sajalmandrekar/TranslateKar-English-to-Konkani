# TranslateKar - English to Konkani (& vice-versa) Language Translator

Developed by:  `Sajal Mandrekar` and `Shreya Deepak Pai`

Dataset generated by: ` Atit Naik, Saylee Phadte, Sajal and Shreya`

A Neural Machine Translator for Konkani to English Translations and vice-versa. It uses the Transformer architecture implemented using tensorflow and keras

## Table of contents

1. [Prerequisite](#Prerequisite)
2. [Test translations using the saved model](#test-translations-using-the-saved-model)
3. [Example Translations](#example-translations)
4. [Evaluation: Bleu Score](#evaluation-bleu-score)
5. [Building BERT Vocabulary](#building-vocabulary)
6. [Training model from scratch](#training-model-from-scratch)
7. [Using Pretrained weights](#using-pretrained-weights)
8. [Terms and Conditions of use](#terms-and-conditions)


## Prerequisite

* Make sure your python version is between 3.8 to 3.11 (to prevent any dependency issues)

* (Optional) Create a virtual environment:
    * `python3 -m venv .myenv`
    * `source ./.myenv/bin/activate`

* Install the libraries using pip: `python3 -m pip install -r requirements.txt`


## Test translations using the saved model

simply run : `python3 run_saved_model.py`

It opens up a prompt to let you select the model (English to Konkani or Konkani to English) or specify the path to the model. On successful loading of the model, you can enter an input and it returns the translated output.

## Example translations

#### English to Konkani (T_BASE_EK_07_07)

Random inputs:
```
source: what is your name?
expected: तुमचें नांव किदें?
predicted: तुमचें नांव कितें ?

source: he likes to play cricket
expected: ताका क्रिकेट खेळपाक आवडटा
predicted: ताका क्रिकेट खेळपाक आवडटा

source: Ramesh is a very kind person
expected: रमेश हो एक बरोच दयाळ मनीस
predicted: रमेश हो एक सामको दयाळू मनीस

source: Goa is my favourite tourist destination
expected: गोंय हें म्हजें आवडीचें पर्यटन थळ
predicted: गोंय हें म्हजें आवडीचें पर्यटन थळ
```

Quotes from the famous :
```
source: Some Quotes from famous people:
predicted: नामनेच्या लोकांचीं कांय कोटीां : १ .

source: ""The only way to do great work is to love what you do."" - Steve Jobs
predicted: "" व्हडलें काम करपाचो एकूच मार्ग म्हणल्यार तुमी जें करतात ताचो मोग करप . ""

source: ""In the end, it's not the years in your life that count. It's the life in your years."" - Abraham Lincoln
predicted: "" शेवटाक , तुमच्या जिवितांत वर्सां न्हय , जीं संख्या . तुमच्या वर्सांनी जिवीत . "" अब्राहम लिंकन

source: ""Success is not final, failure is not fatal: It is the courage to continue that counts."" - Winston Churchill
predicted: "" यशस्वी जावप हें निमाणें न्हय , अपेस घातक न्हय : तें चालूच दवरप हें धैर्य . "" विन्स्टन न्यायालयाक

source: ""It does not matter how slowly you go as long as you do not stop."" - Confucius
predicted: "" जो मेरेन तुमी थांबवपा इतले ल्हवू ल्हवू वतात ताका कसलोच फरक पडना . "" - द्रॅल्फ्लोव्हल

source: ""The greatest glory in living lies not in never falling, but in rising every time we fall."" - Nelson Mandela
predicted: "" जिणेंत सगळ्यांत व्हडलो वैभव केन्नाच पडना , पूण दर खेपे आमी पडटात तेन्ना वाडपाक फट उलयता . "" नेल्सन मंडेला

source: ""The only limit to our realization of tomorrow will be our doubts of today."" - Franklin D. Roosevelt
predicted: फाल्यां आमच्या साक्षात्काराक एकूच मर्यादा म्हळ्यार आयच्या आमचो दुबाव आसतलो . "" - फ्रँकलिन डी .

source: ""Believe you can and you're halfway there."" - Theodore Roosevelt
predicted: "" विस्वास दवरात तुमी शक्य आसात आनी तुमी अर्द्या वाटेर आसात . "" - थिओडोर रूव्हॉल्ट्ट .

source: ""You miss 100% of the shots you don't take."" - Wayne Gretzky
predicted: "" तुमी घेनात ते १०० % शॉट तुमी चुकतात . "" - वेन ग्रेत्झकी

source: ""Don't watch the clock; do what it does. Keep going."" - Sam Levenson
predicted: "" घड्याळ पळोवंक नाकात ; जें चलता तें करात . "" - सॅम लेव्हेनसन
```

#### Konkani to English (T_BASE_KE_17_07)

Random inputs:
```
source: तुमचें नांव कितें?
expected: what is your name?
predicted: What is your name ?

source: ताका क्रिकेट खेळपाक आवडटा
expected: he likes to play cricket
predicted: He likes to play cricket

source: रमेश हो एक बरोच दयाळ मनीस
expected: Ramesh is a very kind person
predicted: Ramesh is a very compassionate person

source: गोंय हें म्हजें आवडीचें पर्यटन थळ
expected: Goa is my favourite tourist destination
predicted: Goa is my favourite tourist destination
```

Miscellaneous inputs:
```
Input: हांव फार्मगुडीच्या गोंय अभियांत्रिकी महाविद्यालयाचो विद्यार्थी
Output: I am a student of Goa Engineering College , farmgudi

Input: हांव संगणक अभियांत्रिकी शिकतां
Output: I am learning computer engineering

Input: मनशाक फकत एकूच गजाल जाय आनी ती तिरस्कार करपा सारकी
Output: A person needs only one thing and that is contemptable

Input: आज रातीं कितें करता?
Output: What does it do tonight ?
```

## Evaluation: Bleu Score

* English to Konkani:
    * model codename: T_BASE_EK_07_07
    * Bleu-4 score: **_29.03%_**

* Konkani to English:
    * model codename: T_BASE_KE_17_07
    * Bleu-4 score: **_23.20%_**


## Building vocabulary

* **This requires you to have a dataset!** The code uses BERT tokenizer (Word-Piece tokenizer) to generated the vocabulary. Note that this is a very CPU/GPU intensive task and thus can take a lot of time depending on your system performance.

* run : `python3 building_vocabulary.py`

* specify the path of your dataset and the max size of the vocabulary

* Generates the vocabulary adding `.vocab` extention to file name of the dataset


## Training model from scratch

* Prerequisites:
    * A parallel corpus in two separate files
    * Two separate vocabulary files for source and target languages

* Modify the configuration file `config.env` to set the dataset paths, vocabulary, epochs and architecture (leave it to default if you want to use the BASE configurations)

* train the model: `python3 transformer_train.py config.env`


## Using Pretrained weights

* open config.env file and modify the variables to specify your dataset file and model name/path (Example shown below):
```
# -----Configurations of the Transformer model----- #

# Model name
MODEL_NAME=TRANS_BASE_EK

## Path to training data of source language
CONTEXT_DATA_PATH=dataset/FULL_DATA.en

## Path to training data of target language
TARGET_DATA_PATH=dataset/FULL_DATA.gom

## Path to vocabulary of source language
CONTEXT_TOKEN_PATH=vocabulary/bert_en.vocab

## Path to vocabulary data of target language
TARGET_TOKEN_PATH=vocabulary/bert_gom.vocab

# Reloading weights from pretrained model (Comment out or leave empty or set to 'None' if not using)
WEIGHTS_PATH=trained_models/T_BASE_EK_07_07/checkpoints/best_model.weights.hdf5
```

* Make sure that architecture variables like `NUM_LAYERS`,`DFF`, etc match the architecture of the pretrained model weights (specified in `config.env` inside the `checkpoints` directory)

* Set the epochs using the `epochs` variable 

* To start training run: `python3 transformer_train.py config.env`


## TERMS AND CONDITIONS

**Disclaimer: Use of this Service and Information**

The following terms and conditions govern your use of this service ("TranslateKar"). By using the Service, you agree to these terms and conditions in full. If you disagree with these terms and conditions or any part of them, you must not use this Service.

**No Liability for Accuracy of Information**

The information provided by this Service is for general informational purposes only. While we strive to provide accurate and up-to-date information, we make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability, or availability with respect to the Service or the information, products, services, or related graphics contained on the Service for any purpose. Any reliance you place on such information is therefore strictly at your own risk.

**No Professional Advice**

The information provided by this Service is not intended as professional advice. You should not rely on the information as an alternative to professional advice. If you have any specific questions about any matter, you should consult a professional.

**No Warranty**

We do not warrant or represent:

1. the completeness or accuracy of the information published on this Service;
2. that the material on this Service is up to date; or
3. that the Service or any service on the Service will remain available.

**Limitations of Liability**

In no event will we be liable for any loss or damage including without limitation, indirect or consequential loss or damage, or any loss or damage whatsoever arising from loss of data or profits arising out of, or in connection with, the use of this Service.

**Links to Other Websites**

Through this Service, you may be able to link to other websites which are not under our control. We have no control over the nature, content, and availability of those sites. The inclusion of any links does not necessarily imply a recommendation or endorse the views expressed within them.
