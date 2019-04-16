import json
import argparse
import sys
import os
import pandas as pd
from io import open

# Takes input and output directories as arguments
parser=argparse.ArgumentParser()
parser.add_argument('--input', default=".", help='The file path of the unzipped DailyDialog dataset')
parser.add_argument('--output', default="./data", help='The file path of the output dataset')
parser.add_argument('--separator', default=r"[TRN]", help='The separator token between context turns')
parser.add_argument('--turns', default="1", help='The number of previous turns to include in the context')
args = parser.parse_args()
INPUT_PATH = args.input
OUTPUT_PATH = args.output
SEPARATOR = args.separator
CONTEXT_LEVEL = int(args.turns)

database_types = ["train", "dev", "test"]

# We make a list of speaker names to anonymise them.
names = []

# Make the output directory if it does not currently exist
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

for database_type in database_types:
    # Open file
    with io.open(INPUT_PATH + "friends_"+ database_type +".json", 'r', encoding="utf-8") as f:
        datastore = json.load(f)

    # Change the encoding of the lines, from cp1252 to utf-8
    def line_encoding_fixer(dialogue_turn):
        dialogue_turn["utterance"] = dialogue_turn["utterance"].encode("utf-8", "replace").decode('cp1252').replace("Â","")

    def line_encoding_fixer_apply(input_list):
        list(map(line_encoding_fixer, input_list))

    # View sample line beforehand
    print(datastore[37][3]["utterance"])

    # Apply the encoding fixer to all bits of dialogue
    list(map(line_encoding_fixer_apply, datastore))

    # Change the lines so that the speaker is recorded as saying each line.
    def line_attributor(dialogue_turn):
        if not dialogue_turn["speaker"] in names:
            names.append(dialogue_turn["speaker"])

        dialogue_turn["utterance"] = "SPKR" + str(names.index(dialogue_turn["speaker"])) + ": " +dialogue_turn["utterance"]

    def line_attributor_apply(input_list):
        list(map(line_attributor, input_list))

    list(map(line_attributor_apply, datastore))

    print(datastore[37][3]["utterance"])

    dialogue_df = pd.DataFrame()

    def dialogue_to_dataframe_context(index_and_input_list_tuple):
        index = index_and_input_list_tuple[0]
        input_list = index_and_input_list_tuple[1]

        dataframe = pd.DataFrame(input_list)

        contexts = pd.Series([""]*len(input_list))
        for i in range(2, 0, -1):
            shifted_dialogue = dataframe.shift(i, fill_value="")["utterance"]

            empty_string_selector = shifted_dialogue == ""
            separator_repeated = [" " + SEPARATOR + " "]*len(input_list)
            separator_repeated_as_df = pd.DataFrame(separator_repeated, columns=["Dialogue"]).mask(empty_string_selector, "")
            separator_repeated_selected = separator_repeated_as_df.mask(empty_string_selector, "")["Dialogue"]

            contexts = contexts + separator_repeated_selected + shifted_dialogue

        dataframe["convo_id"] = index
        dataframe["turn_id"] = list(range(0, len(input_list)))
        dataframe["context"] = contexts.str.replace(pat=str(SEPARATOR), repl="", n=1, regex=False)

        return(dataframe)

    list_of_dialogue_dataframes = list(map(dialogue_to_dataframe_context, enumerate(datastore)))
    dialogue_dataframe = pd.concat(list_of_dialogue_dataframes)

    dialogue_dataframe.to_csv(OUTPUT_PATH+"/"+database_type+".tsv", sep='\t', encoding="utf-8")
