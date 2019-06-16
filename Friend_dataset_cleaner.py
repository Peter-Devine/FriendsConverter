# -*- coding: utf-8 -*-

import json
import argparse
import sys
import os
import pandas as pd
import io

# Takes input and output directories as arguments
parser=argparse.ArgumentParser()
parser.add_argument('--input', default=".", help='The file path of the unzipped DailyDialog dataset')
parser.add_argument('--output', default="./data", help='The file path of the output dataset')
parser.add_argument('--separator', default=r"[TRN]", help='The separator token between context turns')
parser.add_argument('--turns', default="1", help='The number of previous turns to include in the context')
parser.add_argument('--speaker_id', default="False", help='Should each line be preceded by the anonymised speaker id')
parser.add_argument('--is_friends', default="True", help='Is friends dataset (Default True, EmotionPush otherwise)')
args = parser.parse_args()
INPUT_PATH = args.input
OUTPUT_PATH = args.output
SEPARATOR = args.separator
CONTEXT_LEVEL = int(args.turns)
IS_FRIENDS = args.is_friends.upper() == "TRUE"
SPEAKER_ID = args.speaker_id.upper() == "TRUE"

database_types = ["train", "dev", "test"]

# We make a list of speaker names to anonymise them.
names = []

# Make the output directory if it does not currently exist
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

if IS_FRIENDS:
    file_prefix = "friends_"
else:
    file_prefix = "emotionpush_"

for database_type in database_types:
    # Open file
    with io.open(INPUT_PATH + file_prefix + database_type +".json", 'r', encoding="utf-8") as f:
        datastore = json.load(f)

    # Change the encoding of the lines, from cp1252 to utf-8
    def line_encoding_fixer(dialogue_turn):
        dialogue_turn["utterance"] = dialogue_turn["utterance"].encode("utf-8", "replace").decode('ascii', "ignore").replace("\n", "").replace("\t", "")

    def line_encoding_fixer_apply(input_list):
        list(map(line_encoding_fixer, input_list))

    # Apply the encoding fixer to all bits of dialogue
    list(map(line_encoding_fixer_apply, datastore))

    # Change the lines so that the speaker is recorded as saying each line.
    def line_attributor(dialogue_turn):
        if not dialogue_turn["speaker"] in names:
            names.append(dialogue_turn["speaker"])

        if any(word in dialogue_turn["utterance"] for word in names):
            names_in_utterance = set(dialogue_turn["utterance"].split()) & set(names)

            for uttered_name in list(names_in_utterance):
                speaker_id = "SPKR" + str(names.index(uttered_name))
                dialogue_turn["utterance"] = dialogue_turn["utterance"].replace(uttered_name, speaker_id)
       
        if SPEAKER_ID:
            dialogue_turn["utterance"] = "SPKR" + str(names.index(dialogue_turn["speaker"])) + ": " +dialogue_turn["utterance"]

    def line_attributor_apply(input_list):
        list(map(line_attributor, input_list))

    list(map(line_attributor_apply, datastore))

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
