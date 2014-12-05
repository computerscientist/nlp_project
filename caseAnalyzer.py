import bisect
import liblinear
import liblinearutil
import math
import os
import random
import re
import subprocess
import urllib2


historical_relations_to_look_for=[("civil", "right"),
                                  ("worker", "right"),
                                  ("woman", "right"),
                                  ("school", "segregate"),
                                  ("race", "equal"),
                                  ("environment", "regulate"),
                                  ("business", "regulate"),
                                  ("criminal", "right"),
                                  ("prayer", "school"),
                                  ("gay", "right"),
                                  ("national", "interest"),
                                  ("national", "security"),
                                  ("right", "vote"),
                                  ("right", "privacy"),
                                  ("right", "search"),
                                  ("right", "seize"),
                                  ("woman", "equal"),
                                  ("freedom", "speech"),
                                  ("equal", "pay")]


key_words=["dissent",
           "dissents"
           "reverse",
           "reverses",
           "revered",
           "decide",
           "decides",
           "decided",
           "agree",
           "agrees",
           "agreed",
           "disagree",
           "disagrees",
           "disagreed"]


"""
Example of related words list:
["environment", "forest", "forests", "resource", "resources",
 "animal", "animals", "species", "pollution", "toxin", "toxins",
 "cleanup", "clean", "dirty", "contaminate", "contaminated",
 "habitat", "habitats"]
"""


TEXT_DIVIDING_LABEL="--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
JAVA_TAGGER_BASE_DIRECTORY='stanford-postagger-2014-08-27'
ALL_CASE_URLS=None
ALL_CASE_YEARS=None
LIST_OF_APPEARING_WORD_TUPLES=None
SET_OF_APPEARING_WORDS=None


# Tested!
def get_number_of_grammatical_constructs(labeled_grammatical_construct, labeled_text):
    number_of_matching_grammatical_constructs=0
    labeled_grammatical_construct_list=labeled_grammatical_construct.split()
    labeled_text_list=labeled_text.split()

    for index in xrange(0, len(labeled_text_list)-len(labeled_grammatical_construct_list)+1):
        if labeled_grammatical_construct_list[0]==labeled_text_list[index]:
            num_matching_parts=1
            for grammatical_construct_index in xrange(1, len(labeled_grammatical_construct_list)):
                if index+grammatical_construct_index<len(labeled_text_list) and \
                        labeled_grammatical_construct_list[grammatical_construct_index] == \
                        labeled_text_list[index+grammatical_construct_index]:
                    num_matching_parts+=1
                else:
                    break

            if num_matching_parts==len(labeled_grammatical_construct_list):
                number_of_matching_grammatical_constructs+=1

    return number_of_matching_grammatical_constructs


# Tested!
def get_total_number_of_grammatical_constructs(labeled_grammatical_construct_list, labeled_text_list):
    total_number_of_grammatical_constructs=0
    for labeled_grammatical_construct in labeled_grammatical_construct_list:
        total_number_of_grammatical_constructs+=get_number_of_grammatical_constructs(labeled_grammatical_construct,
                                                                                     labeled_text_list)

    return total_number_of_grammatical_constructs


# Tested!
def get_labeled_grammatical_phrases(phrase_input_file):
    labeled_phrases=[]
    f=open(phrase_input_file, 'r')

    current_line=f.readline()
    while(len(current_line)>0):
        labeled_phrases.append(current_line.replace("\n", "").replace("\r", "").strip())
        current_line=f.readline()
    f.close()

    return labeled_phrases


# Tested!
def filter_html(page_html):
    filtered_html=page_html

    # Eliminate special whitespace
    filtered_html=filtered_html.replace('\r', ' ')
    filtered_html=filtered_html.replace('\n', ' ')
    filtered_html=filtered_html.replace('&nbsp;', ' ')

    # Eliminate punctuation and certain special characters
    filtered_html=re.sub(r"[.!:;?(){}\[\],\|~_@^\\]", " ", filtered_html)

    # Eliminate quotation marks
    filtered_html=re.sub(r'"', "", filtered_html)
    filtered_html=re.sub(r"\s+'", " ", filtered_html)
    filtered_html=re.sub(r"'\s+", " ", filtered_html)

    # Deal with special case of single/double quotes at beginning/end of html text (unlikely, but must be considered)
    if filtered_html[0]=="'" or filtered_html[0]=='"':
        filtered_html=filtered_html[1:]
    if filtered_html[-1]=="'" or filtered_html[-1]=='"':
        filtered_html=filtered_html[:-1]

    # Eliminate isolated numbers (to get rid of references) for analysis
    filtered_html=re.sub(r"\s+\d+\s+", " ", filtered_html)

    filtered_html=filtered_html.strip()
    return filtered_html


# Tested!
def get_input_text_from_html_page(URL):
    print URL
    input_text=""
    page_response=urllib2.urlopen(URL)
    page_html=page_response.read()
    page_response.close()
    page_html=filter_html(page_html)

    # Find start and end of body
    try:
        # "'s in "include virtual..." not included due to filter_html algorithm
        case_start=page_html.index('#include virtual = /scripts/includes/caselawheader txt')
        case_start+=len("#include virtual = /scripts/includes/caselawheader txt")
        case_end=page_html.index('#include virtual = /scripts/includes/caselawfooter txt')
    except ValueError:
        # "'s in string to look for not included due to filter_html algorithm
        string_to_look_for="< ------------ END VIEW & PRINT CASES ------------->"
        #string_to_look_for='Jump to: [<a href=#opinion1>Opinion</a>] [<a href=#dissent1>Dissent</a>]<A name=summary1></A>'
        case_start=page_html.index(string_to_look_for)
        case_start+=len(string_to_look_for)
        case_end=page_html.index('< -- END LEFT COLUMN -->')

    inside_element=False
    for index in xrange(case_start, case_end):
        if page_html[index]=='<' and not inside_element:
            inside_element=True
        elif page_html[index]=='>' and inside_element:
            inside_element=False
        elif not inside_element:
            input_text+=page_html[index]

    return input_text


# Tested!
def get_input_text_with_pos(input_text, model='english-left3words-distsim.tagger'):
    f=open('input_text.txt', 'w')
    f.write(input_text)
    f.close()

    pwd=os.getcwd()
    os.chdir(JAVA_TAGGER_BASE_DIRECTORY)
    tagged_text=subprocess.check_output(['./stanford-postagger.sh',
                                         './models/%s' % (model),
                                         '../input_text.txt'])
    os.chdir(pwd)
    os.remove('input_text.txt')

    return tagged_text


# Tested!
def get_number_of_key_word_appearances(input_text, key_word):
    words=input_text.split()
    number_of_appearances=0

    for word in words:
        if word.lower()==key_word.lower():
            number_of_appearances+=1

    return number_of_appearances


# Tested!
def get_total_number_of_key_word_appearances(input_text):
    total_number_of_appearances=0
    for key_word in key_words:
        total_number_of_appearances+=get_number_of_key_word_appearances(input_text, key_word)

    return total_number_of_appearances


# Tested!
def get_total_number_of_relations(input_text, max_distance_threshold=5):
    total_number_of_relations=0
    for relation in historical_relations_to_look_for:
        total_number_of_relations+=get_number_of_specific_relations(relation, input_text, max_distance_threshold)

    return total_number_of_relations


# Tested!
def get_number_of_specific_relations(relation, input_text, max_distance_threshold=5):
    number_of_relations=0
    split_text=input_text.split()
    relation_variation_pairs=get_variation_pairs(relation[0], relation[1])

    for index in range(0, len(split_text)):
        current_subtext=split_text[index:index+max_distance_threshold]
        for pair in relation_variation_pairs:
            if (current_subtext[0]==pair[0] and pair[1] in current_subtext) or \
                   (current_subtext[0]==pair[1] and pair[0] in current_subtext):
                number_of_relations+=1

    return number_of_relations


# Tested!
def get_variations(word):
    if word=="right" or word=="freedom":
        return ["right", "rights", "liberty", "liberties", "freedom",
                "freedoms", "choice", "choices"]
    elif word=="worker":
        return ["worker", "workers", "worker's", "work", "working",
                "employee", "employees", "employee's"]
    elif word=="woman":
        return ["woman", "woman's", "women", "women's", "female", "female's",
                "females", "females'", "girl", "girl's", "girls", "girls'"]
    elif word=="environment":
        return ["environment", "forest", "forests", "resource", "resources",
                "habitat", "habitats", "land", "lands", "river", "rivers",
                "lake", "lakes"]
    elif word=="regulate":
        return ["regulate", "regulation", "regulates", "regulations", "rule", "rules",
                "control", "controls", "oversight", "oversee", "oversees", "monitor",
                "monitors"]
    elif word=="business":
        return ["business", "businesses", "company", "companies", "corporation",
                "corporations", "enterprise", "enterprises"]
    elif word=="criminal":
        return ["criminal", "criminal's", "criminals", "crime", "crimes", "accused",
                "offense", "offenses", "offender", "offender's", "offenders"]
    elif word=="prayer":
        return ["prayer", "prayers", "pray", "prays", "praying"]
    elif word=="school":
        return ["school", "schools"]
    elif word=="segregate":
        return ["segregate", "segregates", "segregation", "separate", "separates",
                "separation", "separations", "divide", "divides", "division", "divisions"]
    elif word=="race":
        return ["race", "races", "color", "colors", "creed", "creeds", "racial",
                "ethnic", "ethnicity", "ethnicities"]
    elif word=="equal":
        return ["equal", "equality", "equivalent", "equivalence"]
    elif word=="gay":
        return ["gay", "gays", "homosexual", "homosexuals", "homosexuality"]
    elif word=="national":
        return ["national", "nationwide", "federal", "nation", "nation's"]
    elif word=="interest":
        return ["interest", "interests", "desire", "desires", "goal", "goals",
                "matter", "matters", "concern", "concerns", "significant",
                "significance", "want", "wants"]
    elif word=="security":
        return ["security", "secure", "safety", "safe", "protect", "protects",
                "protection", "protections", "safeguard", "safeguards", "shield",
                "shields", "defend", "defends", "defense", "surveillance", "guard",
                "guarding", "guards"]
    elif word=="vote":
        return ["vote", "votes", "voting", "suffrage", "poll", "polls", "ballot",
                "ballots"]
    elif word=="privacy":
        return ["privacy", "private", "confidential", "confidentiality"]
    elif word=="search":
        return ["search", "searches", "searching"]
    elif word=="seize":
        return ["seize", "seizes", "seizing", "seizure"]
    elif word=="speech":
        return ["speech", "language", "voice", "voices", "utter", "utters",
                "uttering", "uttered", "utterance", "utterances", "vocalization",
                "vocalizations"]
    elif word=="pay":
        return ["pay", "pays", "paying", "compensation", "compensations",
                "income", "incomes", "wage", "wages"]
    else:
        return [word]


# Tested!
def get_variation_pairs(first_word, second_word):
    first_word_variations=get_variations(first_word)
    second_word_variations=get_variations(second_word)
    variation_pairs=[]

    for fw_variation in first_word_variations:
        for sw_variation in second_word_variations:
            variation_pairs.append((fw_variation, sw_variation))

    return variation_pairs


# Tested!
def get_all_word_variants():
    word_list=["civil", "right", "worker", "woman", "environment", "regulate",
               "business", "criminal", "prayer", "school", "segregate", "race",
               "equal", "gay", "national", "interest", "security", "vote",
               "privacy", "search", "seize", "freedom", "speech", "pay"]

    word_variation_list=[]
    for word in word_list:
        for variation in get_variations(word):
            word_variation_list.append(variation)

    return word_variation_list


# Tested!
def decide_test_data_list(train_data_file, test_data_file):
    train_data_urls=[]
    f=open(train_data_file)
    current_line=f.readline()
    while len(current_line)>0:
        train_data_urls.append(current_line.rsplit(":", 1)[0])
        current_line=f.readline()
    f.close()

    test_data_urls=[]
    f=open(test_data_file)
    test_urls=f.read().split()
    f.close()

    while len(test_data_urls)<1000:
        URL=test_urls[random.randint(0, len(test_urls)-1)]
        if URL not in train_data_urls:
            test_data_urls.append(URL)
            test_urls.remove(URL)

    return test_data_urls


"""
Randomly decide additional training data other than that picked manually.
"""
# Tested!
def decide_additional_training_data_list(manual_train_data_file, url_file):
    manual_train_data_urls=[]
    f=open(manual_train_data_file)
    current_line=f.readline()
    while len(current_line)>0:
        manual_train_data_urls.append(current_line.rsplit(":", 1)[0])
        current_line=f.readline()
    f.close()

    additional_train_data_urls=[]
    f=open(url_file)
    urls=f.read().split()
    f.close()

    while len(additional_train_data_urls)<150:
        URL=urls[random.randint(0, len(urls)-1)]
        if URL not in manual_train_data_urls:
            additional_train_data_urls.append(URL)
            urls.remove(URL)

    return additional_train_data_urls


# Tested!
def find_all_appearing_words(combined_texts):
    global LIST_OF_APPEARING_WORD_TUPLES, SET_OF_APPEARING_WORDS, TEXT_DIVIDING_LABEL
    LIST_OF_APPEARING_WORD_TUPLES=[]
    SET_OF_APPEARING_WORDS=set()
    combined_texts_split=combined_texts.split()

    for word in combined_texts_split:
        if word!=TEXT_DIVIDING_LABEL and word not in SET_OF_APPEARING_WORDS:
            SET_OF_APPEARING_WORDS.add(word)    
            LIST_OF_APPEARING_WORD_TUPLES.append((word, 0))

    LIST_OF_APPEARING_WORD_TUPLES.sort()


# Tested!
def get_bag_of_words(text):
    global SET_OF_APPEARING_WORDS
    word_value_pair_list=LIST_OF_APPEARING_WORD_TUPLES[:]
    words=text.split()
    for word in words:
        index_of_word=bisect.bisect_left(word_value_pair_list, (word, 0))
        word_value_pair_list[index_of_word]=(word_value_pair_list[index_of_word][0], word_value_pair_list[index_of_word][1]+1)

    return word_value_pair_list


# Tested!
def filter_bag_of_words_by_threshold(word_value_pairs, number_of_words, threshold):
    filtered_word_value_pairs=dict()
    filtered_number_of_words=number_of_words
    for word in word_value_pairs:
        if word_value_pairs[word]>=threshold:
            filtered_word_value_pairs[word]=word_value_pairs[word]
        else:
            filtered_number_of_words-=word_value_pairs[word]

    return filtered_number_of_words, filtered_word_value_pairs


# Tested!
def filter_out_pos(tagged_text):
    tagged_text_list=tagged_text.split()
    clean_text=""
    for word in tagged_text_list:
        """
        Have word ending in 's? Tagger will label it separately, so must merge
        it with its parent word just prior.
        """
        if word.split("_")[0]=="'s":
            clean_text="%s%s " % (clean_text[:len(clean_text)-1], word.split("_")[0])
        else:
            clean_text+=word.split("_")[0]+" "

    return clean_text


def form_problem(training_case_file, training_features_file, test_case_file, test_features_file):
    global SET_OF_APPEARING_WORDS
    training_input_text=""
    test_input_text=""

    # Get all training data text combined
    print "Getting training data text..."
    f=open(training_case_file, 'r')
    current_line=f.readline()
    training_labels=[]
    training_case_urls=[]
    while len(current_line)>0:
        training_case_url, training_label=current_line.rsplit(':', 1)
        training_labels.append(training_label.strip())
        training_case_urls.append(training_case_url)
        training_input_text+="%s\n%s\n" % (get_input_text_from_html_page(training_case_url), TEXT_DIVIDING_LABEL)
        current_line=f.readline()
    f.close()
    training_input_text=training_input_text.lower()

    # Get all test data text combined
    print "\nGetting test data text..."
    f=open(test_case_file, 'r')
    current_line=f.readline()
    test_labels=[]
    test_case_urls=[]
    while len(current_line)>0:
        test_case_url, test_label=current_line.rsplit(':', 1)
        test_labels.append(test_label.strip())
        test_case_urls.append(test_case_url)
        test_input_text+="%s\n%s\n" % (get_input_text_from_html_page(test_case_url), TEXT_DIVIDING_LABEL)
        current_line=f.readline()
    f.close()
    test_input_text=test_input_text.lower()

    # Get total bag of appearing words
    find_all_appearing_words(training_input_text+"\n"+test_input_text)

    # Get data features for training data
    f=open('c.txt', 'w')
    f.write(training_input_text)
    f.close()
    labeled_training_input_text=get_input_text_with_pos(training_input_text)#.split("%s_CD" % TEXT_DIVIDING_LABEL)
    labeled_training_input_text=re.sub(r"%s_\S+" % TEXT_DIVIDING_LABEL, TEXT_DIVIDING_LABEL, labeled_training_input_text)
    labeled_training_input_texts=labeled_training_input_text.split(TEXT_DIVIDING_LABEL)
    for current_labeled_training_input_text in labeled_training_input_texts:
        current_labeled_training_input_text=current_labeled_training_input_text.strip()
    training_case_texts=training_input_text.split(TEXT_DIVIDING_LABEL)
    for training_case_text in training_case_texts:
        training_case_text=training_case_text.strip()
    del training_case_texts[-1] # Last "text example" in list just whitespace after last text dividing label
    del labeled_training_input_texts[-1] # Same thing with corresponding labeled version of last "text example"

    g=open(training_features_file, 'w')
    for index in xrange(0, len(training_case_texts)):
        print "Forming training features: %.2f%%" % ((index*100.0)/len(case_texts))
        training_text=training_case_texts[index]
        labeled_training_text=labeled_training_input_texts[index]
        corresponding_url=training_case_urls[index]
        data_features_string=get_data_features_string(training_text, labeled_training_text, corresponding_url)
        g.write("%s %s\n" % (training_labels.pop(0), data_features_string))
    g.close()

    # Get data features for test data now...
    labeled_test_input_text=get_input_text_with_pos(test_input_text)#.split("%s_CD" % TEXT_DIVIDING_LABEL)
    labeled_test_input_text=re.sub(r"%s_\S+" % TEXT_DIVIDING_LABEL, TEXT_DIVIDING_LABEL, labeled_test_input_text)
    labeled_test_input_texts=labeled_test_input_text.split(TEXT_DIVIDING_LABEL)
    for current_labeled_test_input_text in labeled_test_input_texts:
        current_labeled_test_input_text=current_labeled_test_input_text.strip()
    test_case_texts=test_input_text.split(TEXT_DIVIDING_LABEL)
    for test_case_text in test_case_texts:
        test_case_text=test_case_text.strip()
    del test_case_texts[-1] # Last "text example" in list just whitespace after last text dividing label
    del labeled_test_input_texts[-1] # Same thing with corresponding labeled version of last "text example"

    g=open(test_features_file, 'w')
    for index in xrange(0, len(test_case_texts)):
        print "Forming test features: %.2f%%" % ((index*100.0)/len(case_texts))
        test_text=test_case_texts[index]
        labeled_test_text=labeled_test_input_texts[index]
        corresponding_url=test_case_urls[index]
        data_features_string=get_data_features_string(test_text, labeled_test_text, corresponding_url)
        g.write("%s %s\n" % (test_labels.pop(0), data_features_string))
    g.close()


def get_data_features_string(input_text, labeled_input_text, corresponding_url):
    data_features_string=""
    feature_number=1
    tagged_phrases=get_labeled_grammatical_phrases('java_grammatical_phrase_labelings.txt')

    # Look at individual grammatical constructs as features
    for tagged_phrase in tagged_phrases:
        number_found=get_number_of_grammatical_constructs(tagged_phrase, labeled_input_text)
        data_features_string+="%d:%d " % (feature_number, number_found)
        feature_number+=1

    # Look at total number of grammatical constructs as a feature
    total_number_of_grammatical_constructs=get_total_number_of_grammatical_constructs(tagged_phrases, labeled_input_text)
    data_features_string+="%d:%d " % (feature_number, total_number_of_grammatical_constructs)
    feature_number+=1

    # Look at appearances of different key words as featurs
    for key_word in key_words:
        number_of_key_word_appearances=get_number_of_key_word_appearances(input_text, key_word)
        data_features_string+="%d:%d " % (feature_number, number_of_key_word_appearances)
        feature_number+=1

    # Look at total number of key words as a feature
    total_number_of_key_words=get_total_number_of_key_word_appearances(input_text)
    data_features_string+="%d:%d " % (feature_number, total_number_of_key_words)
    feature_number+=1

    # Look at number of appearances of individual two-word relations
    for relation in historical_relations_to_look_for:
        number_of_relation_appearances=get_number_of_specific_relations(relation, input_text, max_distance_threshold=5)
        data_features_string+="%d:%d " % (feature_number, number_of_relation_appearances)
        feature_number+=1

    # Look at total number of all relations found as a feature
    total_number_of_relation_appearances=get_total_number_of_relations(input_text, max_distance_threshold=5)
    data_features_string+="%d:%d " % (feature_number, total_number_of_relation_appearances)
    feature_number+=1

    # Add year of case as a feature
    year_of_case=get_year_of_case(corresponding_url)
    data_features_string+="%d:%d " % (feature_number, year_of_case)
    feature_number+=1

    # Look at bag of words stuff
    word_value_pair_list=get_bag_of_words(input_text) # Assume sorted in alphabetical order
    for word_value_pair in word_value_pair_list:
        data_features_string+="%d:%d " % (feature_number, word_value_pair[1])
        feature_number+=1

    return data_features_string


# Tested!
def get_year_of_case(case_url):
    global ALL_CASE_YEARS, ALL_CASE_URLS
    return int(ALL_CASE_YEARS[ALL_CASE_URLS.index(case_url)])


# Tested!
def get_all_case_urls():
    f=open('dissent/urls.txt', 'r')
    url_list=f.read().split()
    f.close()

    return url_list


# Tested!
def get_list_of_case_years():
    f=open('dissent/years.txt', 'r')
    year_list=f.read().split()
    f.close()

    return year_list


def main():
    global ALL_CASE_YEARS, ALL_CASE_URLS
    ALL_CASE_URLS=get_all_case_urls()
    ALL_CASE_YEARS=get_list_of_case_years()

    # Form model using training data
    #form_problem('Training Cases.txt', 'Training Data Features.txt', 'Test Cases Labeled.txt', 'Test Data Features.txt')
    form_problem('a.txt', 'Training Data Features.txt', 'b.txt', 'Test Data Features.txt')
    training_labels, training_instances = liblinearutil.svm_read_problem('Training Data Features.txt')
    prob = liblinear.problem(training_labels, training_instances)
    model = liblinearutil.train(training_labels, training_instances, '-s 0')

    # Actually test algorithm on test data
    test_labels, test_instances = liblinearutil.svm_read_problem('Test Data Features.txt')
    liblinearutil.predict(test_labels, test_instances, model, "-b 1")


if __name__=="__main__":
    main()
