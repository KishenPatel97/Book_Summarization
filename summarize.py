import stanza, nltk, argparse
from collections import Counter
from string import punctuation
from nltk.corpus import stopwords

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("book", help="file_path to book to be summarized", type=str)

args = parser.parse_args()

# Download and set stopwords
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

# Download English Language Model
stanza.download('en')

# Initiate an English LM stanza pipeline
nlp = stanza.Pipeline('en', processors='tokenize,pos') # we only need tokenize and POS engines


def processBook(fname):
    """ Function to process .txt books, esp. from Gutenberg.
    PARAMS: fname (str) - filepath to book to be processed into a string
    RETURNS: (str) - the string object containing the text
    """
    book = open(fname)      # open book file
    book_lines = []         # initialize storage
    for line in book.readlines():       # iterate through each line
        book_lines.append(line.strip())     # process each line

    return " ".join(book_lines)         # return processed lines

def buildBookSummary(text, limit):
    """Function to read in a text and return a summary made from
    the top (n=limit) sentence constituents
    PARAMS: text (str) - book to be summarized
            limit (int) - the number of the top sentences to use for the summary
    RETURNS: (str) - the summary of the document, represented as the most
                     important sentences by TF-IDF
    """
    # FIND KEYWORDS
    keywords = []       # initialize list of keywords
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']    # list of relevant POS tags
    doc = nlp(text)       # initiate Stanza object from text file
    for sent in doc.sentences:    # iterate through the sentence objects in the document
        for word in sent.words:       # iterate through the word objects in each sentence
            if(word.text in stopWords or word.text in punctuation):   # ignore punctuation and stopwords
                continue
            if word.upos in pos_tag:    # only use words with relevant POS tags: function words aren't key
                keywords.append(word.text)  # append each keyword token for calculating statistics

    # CALCULATE KEYWORD WEIGHTS
    freq_word = Counter(keywords)       # count the keyword token appearances
    max_freq = Counter(keywords).most_common(1)[0][1]     # capture max token occurence for normalization
    for w in freq_word:
        freq_word[w] = (freq_word[w]/max_freq)      # normalize counts

    # CALCULATE SENTENCE WEIGHTS
    sent_strength={}    # initialize dictionary to store sentence weight
    for sent in doc.sentences:      # iterate through sentence objects
        for word in sent.words:       # iterate through word objects
            if word.text in freq_word.keys():     # only consider ketwords in weight calc
                if sent in sent_strength.keys():    # check weight dict to see if sent exists
                    sent_strength[sent.text]+=freq_word[word.text]    # update sentence weight
                else:      # if this is the first time to see this sentence...
                    sent_strength[sent.text]=freq_word[word.text]     # add the word's weight as the initialization of the sentence's weight

    # COLLECT BEST SENTENCES BY WEIGHT
    summary = []  # initialize summary storage
    sorted_x = sorted(sent_strength.items(), key=lambda kv: kv[1], reverse=True) # sort the sentences
    counter = 0     # initialize iteration counter
    for i in range(len(sorted_x)):      # iterate through the sorted list
        summary.append(str(sorted_x[i][0]).capitalize())    # add each BEST SENTENCE consecutively
        counter += 1      # iterate counter
        if(counter >= limit):     # check for stoppage criteria
            break       # stop at sentence limit

    return ' '.join(summary)


if __name__ == "__main__":
    # Get a sample book from Gutenberg
    # !wget "http://www.gutenberg.org/files/11/11-0.txt"
    print(buildBookSummary(processBook(args.book), 15))

