import nltk
import math
import random
import urllib


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


def get_number_of_grammatical_constructs(labeled_grammatical_construct, labeled_text_list):
    number_of_matching_grammatical_constructs=0

    for index in xrange(0, len(labeled_text_list)-len(labeled_grammatical_construct)+1):
        if labeled_grammatical_construct[0] is labeled_text_list[0]:
            num_matching_parts=0
            for grammatical_construct_index in xrange(1, len(labeled_grammatical_construct)):
                if index+grammatical_construct_index<len(labeled_text_list) and \
                        labeled_grammatical_construct[grammatical_construct_index] is \
                        labeled_text_list[index+grammatical_construct_index]:
                    num_matching_parts+=1
                else:
                    break

            if num_matching_parts is len(labeled_grammatical_construct):
                number_of_matching_grammatical_constructs+=1

    return number_of_matching_grammatical_constructs


def get_total_number_of_grammatical_constructs(labeled_grammatical_construct_list, labeled_text_list):
    total_number_of_grammatical_constructs=0
    for labeled_grammatical_construct in labeled_grammatical_construct_list:
        total_number_of_grammatical_constructs+=get_number_of_grammatical_constructs(labeled_grammatical_construct,
                                                                                     labeled_text_list)

    return total_number_of_grammatical_constructs


def get_pos_tags_of_grammatical_phrases(phrase_input_file):
    list_of_tagged_phrases=[]
    f=open(phrase_input_file, 'r')
    current_phrase=f.readline()

    while len(current_phrase)>0:
        tokens = nltk.word_tokenize(current_phrase)
        tagged = nltk.pos_tag(tokens)
        list_of_tagged_phrases.append(tagged)
        current_phrase=f.readline()

    f.close()
    return list_of_tagged_phrases


def get_input_text_from_html_page(URL):
    input_text=""
    page_response=urllib2.urlopen(URL)
    page_html=page_response.read()
    page_response.close()

    # Find start and end of body
    case_start=page_html.index('U.S. Supreme Court')
    case_end=page_html.index('<!-- #include virtual =')

    inside_element=False
    for index in xrange(case_start, case_end):
        if page_html['index']=='<' and not inside_element:
            inside_element=True
        elif page_html['index']=='>' and inside_element:
            inside_element=False
        elif not inside_element:
            input_text+=page_html[index]

    return input_text


def get_input_text_with_pos(input_text):
    tokens = nltk.word_tokenize(input_text)
    tagged_word_list = nltk.pos_tag(tokens)

    return tagged_word_list


def get_number_of_key_word_appearances(input_text, key_word):
    words=input_text.split()
    number_of_appearances=0

    for word in words:
        if word is key_word:
            number_of_appearances+=1

    return number_of_appearances


def get_total_number_of_key_word_appearanes(input_text):
    total_number_of_appearances=0
    for key_word in key_words:
        total_number_of_appearances+=get_number_of_key_word_appearances(input_text, key_word)

    return total_number_of_appearances


def get_total_number_of_relations(input_text, max_distance_threshold=5):
    split_text=input_text.split()
    total_number_of_relations=0

    for relation in historical_relations_to_look_for:
        variation_pairs=get_variation_pairs(relation[0], relation[1])
        for index in range(0, len(input_text)-max_distance_threshold+1):
            current_subtext=input_text[index:index+max_distance_threshold]
            for pair in variation_pairs:
                if pair[0] in current_subtext and pair[1] in current_subtext:
                    total_number_of_relations+=1

    return total_number_of_relations

        
def get_number_of_specific_relations(relation, input_text, max_distance_threshold=5):
    number_of_relations=0
    split_text=input_text.split()
    relation_variation_pairs=get_variation_pairs(relation[0], relation[1])

    for index in range(0, len(input_text)-max_distance_threshold+1):
        current_subtext=input_text[index:index+max_distance_threshold]
        for pair in relation_variation_pairs:
            if pair[0] in current_subtext and pair[1] in current_subtext:
                number_of_relations+=1

    return number_of_relations


def get_variations(word):
    if word is "right" or word is "freedom":
        return ["right", "rights", "liberty", "liberties", "freedom",
                "freedoms", "choice", "choices"]
    elif word is "worker":
        return ["worker", "workers", "worker's", "work", "working",
                "employee", "employees", "employee's"]
    elif word is "woman":
        return ["woman", "woman's", "women", "women's", "female", "female's",
                "females", "females'", "girl", "girl's", "girls", "girls'"]
    elif word is "environment":
        return ["environment", "forest", "forests", "resource", "resources",
                "habitat", "habitats", "land", "lands", "river", "rivers",
                "lake", "lakes"]
    elif word is "regulate":
        return ["regulate", "regulation", "regulates", "regulations", "rule", "rules",
                "control", "controls", "oversight", "oversee", "oversees", "monitor",
                "monitors"]
    elif word is "business":
        return ["business", "businesses", "company", "companies", "corporation",
                "corporations", "enterprise", "enterprises"]
    elif word is "criminal":
        return ["criminal", "criminal's", "criminals", "crime", "crimes", "accused",
                "offense", "offenses", "offender", "offender's", "offenders"]
    elif word is "prayer":
        return ["prayer", "prayers", "pray", "prays", "praying"]
    elif word is "school":
        return ["school", "schools"]
    elif word is "segregate":
        return ["segregate", "segregates", "segregation", "separate", "separates",
                "separation", "separations", "divide", "divides", "division", "divisions"]
    elif word is "race":
        return ["race", "races", "color", "colors", "creed", "creeds", "racial",
                "ethnic", "ethnicity", "ethnicities"]
    elif word is "equal":
        return ["equal", "equality", "equivalent", "equivalence"]
    elif word is "gay":
        return ["gay", "gays", "homosexual", "homosexuals", "homosexuality"]
    elif word is "national":
        return ["national", "nationwide", "federal", "nation", "nation's"]
    elif word is "interest":
        return ["interest", "interests", "desire", "desires", "goal", "goals",
                "matter", "matters", "concern", "concerns", "significant",
                "significance", "want", "wants"]
    elif word is "security":
        return ["security", "secure", "safety", "safe", "protect", "protects",
                "protection", "protections", "safeguard", "safeguards", "shield",
                "shields", "defend", "defends", "defense", "surveillance", "guard",
                "guarding", "guards"]
    elif word is "vote":
        return ["vote", "votes", "voting", "suffrage", "poll", "polls", "ballot",
                "ballots"]
    elif word is "privacy":
        return ["privacy", "private", "confidential", "confidentiality"]
    elif word is "search":
        return ["search", "searches", "searching"]
    elif word is "seize":
        return ["seize", "seizes", "seizing", "seizure"]
    elif word is "speech":
        return ["speech", "language", "voice", "voices", "utter", "utters",
                "uttering", "uttered", "utterance", "utterances", "vocalization",
                "vocalizations"]
    elif word is "pay":
        return ["pay", "pays", "paying", "compensation", "compensations",
                "income", "incomes", "wage", "wages"]
    else:
        return [word]


def get_variation_pairs(first_word, second_word):
    first_word_variations=get_variations(first_word)
    second_word_variations=get_variations(second_word)
    variation_pairs=[]

    for fw_variation in first_word_variations:
        for sw_variation in second_word_variations:
            variation_pairs.append((fw_variation, sw_variation))

    return variation_pairs


def print_all_word_variants():
    word_list=["civil", "right", "worker", "woman", "environment", "regulate",
               "business", "criminal", "prayer", "school", "segregate", "race",
               "equal", "gay", "national", "interest", "security", "vote",
               "privacy", "search", "seize", "freedom", "speech", "pay"]

    variation_list_string=""
    for word in word_list:
        for variation in get_variations(word):
            variation_list_string+=variation+" "

    print variation_list_string

    
def decide_test_data_list(train_data_file, test_data_file):
    train_data_urls=[]
    f=open(train_data_file)
    current_line=f.readline()
    while len(current_line)>0:
        train_data_urls.append(current_line.split(":")[0])
        current_line=f.readline()
    f.close()

    test_data_urls=[]
    f=open(test_data_file)
    test_urls=f.read().split()
    f.close()

    while len(test_data_urls)<2000:
        URL=test_urls[random.randint(0, len(test_urls)-1)]
        if URL not in train_data_urls:
            test_data_urls.append(URL)
            test_urls.remove(URL)

    return test_data_urls


"""
Randomly decide additional training data other than that picked manually.
"""
def decide_additional_training_data_list(manual_train_data_file, url_file):
    manual_train_data_urls=[]
    f=open(manual_train_data_file)
    current_line=f.readline()
    while len(current_line)>0:
        manual_train_data_urls.append(current_line.split(":")[0])
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


def get_bag_of_words(text):
    word_value_pairs=dict()
    words=text.split()
    for word in words:
        if word not in word_value_pairs:
            word_value_pairs[word]=1
        else:
            word_value_pairs[word]+=1

    return len(words), word_value_pairs


def filter_bag_of_words_by_threshold(word_value_pairs, threshold):
    for word in word_value_pairs:
        if word_value_pairs[word]<threshold:
            del word_value_pairs[word]

        
def get_probability_of_word(word, word_value_pairs, number_of_words):
    if word not in word_value_pairs:
        return 0  # Set to some really small value instead?
    else:
        return word_value_pairs[word]/number_of_words


"""
Calculate the combined log probability of all words in a piece of text.
This only calculates the log probability of the words combined: the final
log probability used elsewhere in the program must also add in the log
probability of the class (e.g. agree/disagree) that a particular case is in.
"""
def calculate_log_probability(word_value_pairs, number_of_words):
    logprob=0
    for word in word_value_pairs:
        logprob+=get_probability_of_word(word, word_value_pairs, number_of_words)

    return logprob


def filter_out_pos(tagged_text):
    tagged_text_list=tagged_text.split()
    clean_text=""
    for word in tagged_text_list:
        """
        Have word ending in 's? Tagger will label it separately, so must merge
        it with its parent word just prior.
        """
        if word.split("_")[0]=="'s":
            clean_text=clean_text[:len(clean_text-1)]+word.split("_")[0]+" "
        else:
            clean_text+=word.split("_")[0]+" "

    return clean_text


def form_problem(training_case_file, training_data_features_file):
    f=open(training_case_file, 'r')
    g=open(training_data_features_file, 'w')

    current_line=f.readline()
    while len(current_line)>0:
        case_url, label=current_line.split(':')
        input_text=get_input_text_from_html_page(current_line.split(':')[0])
        data_features_string=get_data_features_string(input_text)
        g.write("%s %s\n" % (label, data_features_string))
        current_line=f.readline()

    g.close()
    f.close()


def get_data_features_string(input_text):
    # get_bag_of_words, filter_bag_of_words_by_threshold, calculate_log_probability
    data_features_string=""
    feature_number=1
    tagged_phrases=get_pos_tags_of_grammatical_phrases('grammatical_phrase_labelings.txt')
    labeled_text_list=get_input_text_with_pos(input_text)

    # Look at individual grammatical constructs as features
    for tagged_phrase in tagged_phrases:
        number_found=get_number_of_grammatical_constructs(labeled_grammatical_construct, labeled_text_list)
        data_features_string+="%d:%d" % (feature_number, number_found)
        feature_number+=1

    # Look at total number of grammatical constructs as a feature
    total_number_of_grammatical_constructs=get_total_number_of_grammatical_constructs(tagged_phrases, labeled_text_list)
    data_features_string+="%d:%d" % (feature_number, total_number_of_grammatical_constructs)
    feature_number+=1

    # Look at appearances of different key words as featurs
    for key_word in key_words:
        number_of_key_word_appearances=get_number_of_key_word_appearances(input_text, key_word)
        data_features_string+="%d:%d" % (feature_number, number_of_key_word_appearances)
        feature_number+=1

    # Look at total number of key words as a feature
    total_number_of_key_words=get_total_number_of_key_word_appearanes(input_text)
    data_features_string+="%d:%d" % (feature_number, total_number_of_key_words)
    feature_number+=1


    get_total_number_of_relations(input_text, max_distance_threshold=5)
    get_number_of_specific_relations(relation, input_text, max_distance_threshold=5)

    number_of_words, word_value_pairs=get_bag_of_words(input_text)
    filtered_bag_of_words=filter_bag_of_words_by_threshold(word_value_pairs, 5)
    log_probability=log_calculate_log_probability(word_value_pairs, number_of_words)
    data_features_string+="%d:%d" % (feature_number, log_probability)
    feature_number+=1

    return data_features_string


def main():
    # print decide_additional_training_data_list('Training Cases.txt', 'urls.txt')
    # print get_pos_tags_of_grammatical_phrases('short grammatical phrases to look for')
    form_problem('Training Cases.txt', 'Training Data Features.txt')
    labels, instances = svm_read_problem('Training Data Features.txt')
    prob = problem(labels, instances)
    model = train(labels, instances, '-s 0')
    predict(labels, instances, model, "-b 1")


if __name__=="__main__":
    main()
